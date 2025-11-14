from typing import List, Dict

import cpmpy as cp
from cpmpy.expressions.utils import eval_comparison
from cpmpy.expressions.core import BoolVal, Operator, Comparison, Expression
from cpmpy.expressions.variables import _BoolVarImpl, _NumVarImpl
from cpmpy.solvers.pumpkin import CPM_pumpkin
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.expressions.python_builtins import any as cpm_any, all as cpm_all
from cpmpy.transformations.get_variables import get_variables
import gzip

import re
from .utils import normalize
from tqdm import tqdm

RE_PROP = (r"^i\s+(?P<id>[1-9]\d*)\s+"
           r"(?P<premises>(?:-?[1-9]\d*\s*)*)"
           r"(?:0\s+(?P<propagated>-?[1-9]\d*)?\s*)?"
           r"(?:c:(?P<tag>-?[1-9]\d*))?\s*"
           r"(?:l:(?P<filtering_algorithm>\w+))?$")

PROP_REQUIRED = ['id', 'premises', 'tag']

# format: n <step_id> <atomic constraint ids> [0 <propagation hint>]
RE_NOGOOD = re.compile(r"^n\s+(?P<id>-?\d+)\s+"
                       r"(?:(?P<lit_ids>(?:-?\d+\s*)*?)0\s+)?"
                       r"(?P<hint>(?:-?\d+\s*)*)")


def get_clause(premises, propagated):
    if propagated is None:
        return sorted([-i for i in premises], key=abs)
    else:
        return sorted([-i for i in premises] + [int(propagated)], key=abs)


class PumpkinProofParser(CPM_pumpkin):


    def __init__(self, model, **kwargs):
        self.user_cons = dict()
        super().__init__(cpm_model=model, **kwargs)

    def solve(self, proof=None, **kwargs):
        res = super().solve(prove=proof is not None,
                            proof_name=proof,
                            **kwargs)
        if proof is not None:
            self.prefix = proof
        return res

    VAR_NAMES_REGEX = re.compile("[^0-9a-zA-Z]+")
    def solver_var(self, cpm_var):
        # if isinstance(cpm_var, _NumVarImpl):
            # we need to replace variable names to those supported by the proof format
            # cpm_var.name = re.sub(self.VAR_NAMES_REGEX, "_", cpm_var.name)
        return super().solver_var(cpm_var)

    def __add__(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """
        from pumpkin_solver import constraints

        if not hasattr(self, "tags"): self.tags = dict()  # mapping from tag -> solver constraint
        if not hasattr(self, "user_cons"): self.used_cons = dict()  # mapping from solver constraint -> user constraint

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for orig_expr in set(toplevel_list(cpm_expr_orig, merge_and=True)):
            for cpm_expr in self.transform(orig_expr):
                # save tag
                tag = self.pum_solver.new_constraint_tag()
                self.tags[int(tag)] = cpm_expr
                self.user_cons[cpm_expr] = orig_expr

                if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":  # found implication
                    bv, subexpr = cpm_expr.args
                    for cons in self._get_constraint(subexpr, tag=tag):
                        self.pum_solver.add_implication(cons, self.solver_var(bv))
                else:
                    solver_constraints = self._get_constraint(cpm_expr, tag=tag)

                    for cons in solver_constraints:
                        self.pum_solver.add_constraint(cons)
        return self

    def read_proof_tree(self, prefix=None):
        """
            Parses the proof-log into memory.
            Returns a list of inference steps and nogoods steps
        """

        if prefix is None:
            prefix = self.prefix
        assert prefix is not None, "solver has to be ran with proof-logging enabled, or prefix has to be user-supplied"

        proof = []
        nogood_ids = dict()
        literals = dict()
        repeated_nogoods = dict()

        with (gzip.open(prefix, "rb") as f):
            for line_nb, line in enumerate(tqdm(f.readlines(), desc="Parsing proof")):
                line = line.decode("utf-8")
                # print(line[:-1])
                if line[0] == "a":  # atomic constraint literal
                    a, lit_id, *lit_str = line.split(" ")
                    lit = self.parse_one_lit(" ".join(lit_str))
                    literals[int(lit_id)] = lit

                elif line[0] == "i":  # found inference
                    # premises & cons -> propagated
                    # rewrite to cons -> ~cp.all(premises) \/ implied_lit
                    match = re.match(RE_PROP, line).groupdict()
                    lit_ids = [int(id) for id in match['premises'].split()]
                    cpm_lits = map(lambda id: literals[abs(id)], lit_ids)
                    premises = normalize([lit if id > 0 else ~lit for id, lit in zip(lit_ids, cpm_lits)])

                    if match['propagated'] is None:  # found an "inference no-good"
                        propagated_lits = []
                        propagated = []
                    else:
                        propagated_lits = [int(match['propagated'])]
                        lit = propagated_lits[0]
                        propagated = normalize([literals[lit] if lit > 0 else ~literals[abs(lit)]])
                        assert len(propagated) == 1, f"Propagated should be a single literal, got {propagated}"

                    derived_clause = cpm_any(normalize([~lit for lit in premises] + propagated))


                    if match['filtering_algorithm'] == "nogood":  # nogood repeated from before, maybe slightly altered
                        # can be repeated nogood or just clausal propagator
                        tag = int(match['tag'])
                        if tag in nogood_ids and tag < int(match['id']):
                            # is it a repeat of a nogood?
                            prev_nogood = proof[nogood_ids[tag]]
                            int_clause = frozenset([-i for i in lit_ids] + propagated_lits)
                            if int_clause == prev_nogood["int_clause"]:
                                # no point in repeating... just use the previous nogood directly
                                repeated_nogoods[int(match['id'])] = tag
                                # proof.append(dict(
                                #     id = int(match['id']),
                                #     type = "inference",
                                #     reasons = [int(match['tag'])],
                                #     derived = [derived_clause]
                                # ))
                                continue

                        proof.append(dict(
                            id = int(match['id']),
                            type="inference",
                            reasons = [self.tags[int(match['tag'])]],
                            derived = [derived_clause]
                        ))
                    elif match['filtering_algorithm'] == "initial_domain": # dummy literals to make proof valid
                        proof.append(dict(
                            id = int(match['id']),
                            type="inference",
                            reasons = [],
                            derived = [derived_clause]
                        ))
                    elif match['tag'] is None:
                        raise ValueError(f"Expected step to be tagged but got {line}")
                    else:
                        cons = self.tags[int(match['tag'])]
                        proof.append(dict(
                            id = int(match['id']),
                            type="inference",
                            reasons = [cons],
                            derived = [derived_clause]
                        ))

                elif line[0] == "n":  # found nogood
                    # format = [premises] -> False
                    # so, we rewrite as ~all(premises)
                    match = re.match(RE_NOGOOD, line).groupdict()

                    lit_ids = [int(id) for id in match['lit_ids'].strip().split()]

                    cpm_lits = map(lambda id: literals[abs(id)], lit_ids)
                    nogood = [lit if id > 0 else ~lit for id, lit in zip(lit_ids, cpm_lits)]

                    if len(nogood) == 0:
                        nogood = cp.BoolVal(False)
                    else:
                        nogood = cpm_any(normalize([~lit for lit in nogood]))

                    reasons = list(map(int, match['hint'].strip().split()))
                    reasons = [repeated_nogoods.get(id,id) for id in reasons] # skip repeated nogoods

                    proof.append(dict(
                        id = int(match['id']),
                        type="nogood",
                        reasons = reasons,
                        derived = [nogood],
                        int_clause = frozenset([-i for i in lit_ids])
                    ))

                    nogood_ids[int(match['id'])] = len(proof)-1 # map to index of proof

                elif line[0] == "c":  # conclusion
                    assert line == "c UNSAT\n", "Currently only support proof of unsatisfiability"
                    break

        return proof


    def parse_one_lit(self, string):
        """
            Parse one literal
            A literal is a comparison of a variable with an integer.
            Allowed comparisons are "==", "!=", "<=", ">="
        """
        string = string.strip().strip(" []")

        # TODO: not sure if this is still allowed in new proof format
        if "true" in string: return True

        # find out which comparison
        for comp in ("!=", "<=", ">=", "=="):
            if comp in string: break
        else:
            raise ValueError(f"Expected comparison but got {string}")

        lhs, rhs = string.split(comp)
        lhs, rhs = lhs.strip(), int(rhs.strip())
        # find variable
        for var in self._varmap:
            if var.name == lhs:
                return eval_comparison(comp, var, rhs)

        # false literal can be encoded as int != int
        try:
            lhs = int(lhs)
            return eval_comparison(comp, lhs, rhs)
        except ValueError:
            raise ValueError(f"Unknown lhs of comparison in literal {string}")
