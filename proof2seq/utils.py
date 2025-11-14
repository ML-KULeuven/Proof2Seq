import numpy as np

import cpmpy as cp

from cpmpy.expressions.core import Expression, Operator, Comparison, BoolVal
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl
from cpmpy.expressions.utils import is_num

from cpmpy.transformations.get_variables import get_variables as cpm_get_variables
from cpmpy.transformations.negation import push_down_negation
from cpmpy.expressions.utils import flatlist



def to_list_recursive(lst):
    if hasattr(lst, '__iter__'):
        return [to_list_recursive(v) for v in lst]
    return lst

def get_variables(constraints):
    """
    Helper function to get variables in collection of constraints
    Accepts any (nested) iterator as input
    """
    return cpm_get_variables(to_list_recursive(constraints))

def normalize(lst_of_exprs):
    """
        Normalize a list of CPMpy expressions.
        Will transform any < to <= and > to >=.
        Output is guaranteed to be a list Comparisons or BoolVal (no NegBoolView or _BoolVarImpl)
    """
    newlst = []
    for cpm_expr in push_down_negation(lst_of_exprs):

        if isinstance(cpm_expr, Operator) and cpm_expr.name == "or":
            newlst.append(cp.any(normalize(cpm_expr.args)))

        elif isinstance(cpm_expr, NegBoolView):
            newlst.append(cpm_expr._bv <= 0)

        elif isinstance(cpm_expr, _BoolVarImpl):
            newlst.append(cpm_expr >= 1)

        elif isinstance(cpm_expr, BoolVal):
            newlst.append(cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            assert isinstance(lhs, _NumVarImpl) and is_num(rhs), f"Expected comparison to be canonical, but got {cpm_expr}"
            if cpm_expr.name == "<":
                newlst.append(lhs <= rhs - 1)
            elif cpm_expr.name == ">":
                newlst.append(lhs >= rhs + 1)
            else:  # other comparisons are fine
                newlst.append(cpm_expr)
        else:
            raise ValueError(f"Unexpected expression: {cpm_expr}")

    return sorted(newlst, key=str)  # , key=lambda x : x.args[0].name)

def get_cpm_reasons(step, proof):

    if isinstance(proof, list):
        proof = {step['id'] for step in proof}

    cpm_reasons = []
    for used in step['reasons']:
        if isinstance(used, int): # it's a step id
            cpm_reasons.append(proof[used]['derived'])
        else:
            assert isinstance(used, Expression), f"Expected int or expression but got {used}"
            cpm_reasons.append(used)
    return cpm_reasons

def sanity_check_proof(proof):

    proof_dict = {step['id'] : step for step in proof}
    for step in proof:
        # check all used steps have a smaller id
        for r in step['reasons']:
            if isinstance(r, int) and r >= step['id']:
                raise ValueError(f"Proof step {step} uses steps that occur later in the proof!\n{step['reasons']}")

        # check output is logically implied by input
        cpm_reasons = get_cpm_reasons(step, proof_dict)
        cpm_derived = step['derived']

        for cons in cpm_derived:
            if not isinstance(cons, Expression):
                raise ValueError(f"Expected expression, got {cons} in proof step {step}")


        unsat_model = cp.Model(cpm_reasons + [~cp.all(cpm_derived)])
        if unsat_model.solve() is not False:
            raise ValueError(f"Error in proof step with id {step['id']}!\n"
                             f"Reasons do not logically imply derived constraint!\n"
                             f"Reasons:\n\t" +'\n\t'.join(map(str,cpm_reasons)) +"\n"
                             f"Derived: {cpm_derived}")


def get_proof_statistics(proof):
    n_reasons = [len(step['reasons']) for step in proof]
    return dict(
        length = len(proof),
        avg_reasons = sum(n_reasons) / len(n_reasons),
        std_reasons = np.std(n_reasons),
        max_reasons = max(n_reasons)
    )

def print_proof_statistics(proof, name="Proof", precision=2):

    print(f"Statistics for {name}:")
    stats = get_proof_statistics(proof)
    print("#steps:", stats['length'], end="\t")
    print("avg #reasons:", round(stats['avg_reasons'],precision), end="\t")
    print("std #reasons:", round(stats['std_reasons'], precision), end="\t")
    print("max #reasons:", stats['max_reasons'], end="\t")
    print("\n")


def get_sequence_statistics(sequence):

    n_cons = [len(step['constraints']) for step in sequence]
    return dict(
        length = len(sequence),
        avg_cons = sum(n_cons) / len(sequence),
        std_cons = np.std(n_cons),
        max_cons = max(n_cons)
    )

def print_sequence_statistics(sequence, precision=2):

    print("Statistics for explanation sequence:")
    stats = get_sequence_statistics(sequence)
    print("#steps:", stats['length'], end="\t")
    print("avg #constraints:", round(stats['avg_cons'],precision), end="\t")
    print("std #constraints:", round(stats['std_cons'], precision), end="\t")
    print("max #constraints:", stats['max_cons'], end="\t")
    print("\n")



def get_domains_from_literals(literals):
    domains = dict()
    for lit in literals:
        if isinstance(lit, BoolVal):
            if lit.value() is False:
                return [lit]
        elif isinstance(lit, Comparison):
            var, val = lit.args
            assert is_num(val), f"Expected atomic constraint but got {lit}"
            if var not in domains:
                domains[var] = set(range(var.lb, var.ub + 1))
            if lit.name == "==":
                domains[var] &= {val}
            elif lit.name == "!=":
                domains[var] -= {val}
            elif lit.name == ">=":
                domains[var] -= set(range(var.lb, val))
            elif lit.name == ">":
                domains[var] -= set(range(var.lb, val + 1))
            elif lit.name == "<=":
                domains[var] -= set(range(val + 1, var.ub + 1))
            elif lit.name == "<":
                domains[var] -= set(range(val, var.ub + 1))
            else:
                raise ValueError(f"Unexpected comparison {lit.name}")
        else:
            raise ValueError(f"Unexpected literal {lit}")
    return domains

def minimize_literals(literals):

    domains = get_domains_from_literals(literals)
    new_literals = []
    for var, dom in domains.items():
        if len(dom) == 1:
            new_literals.append(var == next(iter(dom)))
        elif len(dom) == 0:
            return [cp.BoolVal(False)]
        else:
            new_literals.append(var >= min(dom))
            for i in range(min(dom), max(dom)+1):
                if i not in dom:
                    new_literals.append(var != i)
            new_literals.append(var <= max(dom))

    return new_literals

def pretty_print_sequence(sequence, indent=0, format="domain"):

    for i, step in enumerate(sequence):
        lines = []
        if format == "literals":
            lines += ["    Input literals:"]
            lines += ["        "+",".join(map(str, step['input_lits']))]
        elif format == "minliterals":
            lines += ["    Input literals:"]
            lines += ["        "+",".join(map(str, minimize_literals(step['input_lits'])))]
        elif format == "domain":
            lines += ["    Input domains:"]
            domains = get_domains_from_literals(step['input_lits'])
            parts = []
            for var in get_variables(step['constraints']):
                if var in domains:
                    parts.append(f"{var} ∈ {sorted(domains[var])}")
                else:
                    parts.append(f"{var} ∈ [{var.lb}..{var.ub}]")
            lines += ["         " +", ".join(parts)]
        else:
            raise ValueError(f"Expected 'literals' or 'domain' as format but got {format}")

        lines += ["    Constraints:"]
        lines += ["        "+str(c) for c in sorted(step['constraints'], key=str)]

        lines += ["    Output literals:"]
        lines += ["        "+",".join(map(str, step['output_lits']))]

        width = max(len(l) for l in lines)
        print("--",i+1,"-"*(width+1-len(str(i+1))), sep="")
        for line in lines:
            print("    "*indent,"|", line," "*(width-len(line)+1),"|", sep="")
        print("-"*(width+3))





