import copy
from time import time

import cpmpy as cp
from cpmpy.expressions.core import Operator, Comparison
from cpmpy.expressions.utils import is_num
from cpmpy.expressions.variables import _NumVarImpl
from cpmpy.solvers.solver_interface import ExitStatus

from .parsing import PumpkinProofParser
from .simplify import simplify_proof
from .minimize import minimize_proof

from .utils import sanity_check_proof, get_variables, print_proof_statistics, pretty_print_sequence, \
    print_sequence_statistics, minimize_literals


def compute_sequence(model,
                     minimization_phase1 = "proof",
                     minimization_phase2 = "global",
                     mus_solver = "exact",
                     mus_type="smus",
                     proof_name="proof.gz",
                     do_sanity_check = True,
                     verbose = 0,
                     pumpkin_solver=None,
                     time_limit=3600,
                     ):
    """
        Compute a step-wise explanation sequence by starting from a DRCP proof.
        The pipeline consists of the following steps
        1. Solve model and parse proof
        2. Remove steps with auxiliary variables
        4. Replace solver-level constraints with user-level constraints
        6. [Optional] Minimize reasons in the proof
        5. Remove steps deriving clauses over multiple variables
        6. [Optional] Minimize reasons in the proof

    :param model: Unsatisfiable CPMpy model
    :param minimization_phase1: Which type of proof minimization we want to do in the first phase
                                Can be any of:
                                    proof: don't do anything
                                    trim: trim the proof based on the reasons in the proof
                                    mus: trim the proof using MUS-minimization
                                    smus: trim the proof using Smallest-MUS minimization
    :param minimization_phase2: Which type of proof minimization we want to do in the second phase
                                Can be any of:
                                    proof: don't do anything
                                    trim: trim the proof based on the reasons in the proof
                                    mus: trim the proof using MUS-minimization
                                    smus: trim the proof using Smallest-MUS minimization
    :param mus_solver: which solver to use during MUS/SMUS minimization
    :param mus_type: which type of MUS to use: deletion-based MUS or SMUS
    :param proof_name: the name of the proof stored on disk
    :param do_sanity_check: For debugging, check whether the proof is valid
    :param verbose: set verbosity and print statistics of proof
    :param pumpkin_solver: CPMpy Pumpkin solver, only needed to run experiments, to ensure proof is re-used among configs
    :return: sequence of explanation steps
    """
    start = time()
    if pumpkin_solver is None:
        solver = PumpkinProofParser(model)
        assert solver.solve(proof=proof_name, time_limit=time_limit) is False
        if solver.status().exitstatus == ExitStatus.UNKNOWN:
            raise TimeoutError
    else:
        solver = pumpkin_solver

    if verbose > 0:
        print(f"Took {solver.status().runtime}seconds to solve model and produce proof")

    # read the proof from disk
    proof = solver.read_proof_tree()
    if do_sanity_check: sanity_check_proof(proof)
    if verbose > 0:
        print_proof_statistics(proof, "initial proof")

    # Remove steps deriving information about auxiliary variables
    user_vars = frozenset(get_variables(model.constraints))
    def only_user_vars(step):
        return frozenset(get_variables(step['derived'])) <= user_vars

    proof = simplify_proof(proof, condition=only_user_vars)
    if do_sanity_check: sanity_check_proof(proof)
    if verbose > 0:
        print_proof_statistics(proof, "proof without auxiliary variables")

    # Replace solver-level constraints with user-level constraints
    for step in proof:
        step['reasons'] = [solver.user_cons.get(r,r) for r in step['reasons']]

    # There are a lot of repeated inferences in the proof,
    # but we don't care about the exact inferences made, just about the constraint that was used.
    inference_map = dict() # id to constraint
    newproof = []
    for step in proof:
        if step['type'] == "inference":
            if len(step['reasons']) == 0: # initial domain
                step['reasons'] = [cp.BoolVal(True)]
            assert len(step['reasons']) == 1, f"Inferences should have a single constraint as reason but got {step}"
            assert isinstance(step['reasons'][0], Expression), f"Inferences should have a single constraint as reason but got {step}"
            inference_map[step['id']] = step['reasons'][0]
        elif step['type'] == 'nogood':
            new_reasons = set(inference_map.get(r,r) for r in step['reasons']) - {cp.BoolVal(True)}
            step = copy.copy(step)
            step['reasons'] = list(new_reasons)
            newproof.append(step)

    proof = newproof

    assert all(step['type'] == 'nogood' for step in newproof)

    # Do the first minimization phase
    time_limit -= (time() - start)
    proof = minimize_proof(proof, model,
                           minimization_type=minimization_phase1,
                           mus_type=mus_type, mus_solver=mus_solver,
                           verbose=verbose, time_limit=time_limit)
    if do_sanity_check: sanity_check_proof(proof)
    if verbose > 0:
        print_proof_statistics(proof, "proof after first minimization phase")

    # Remove steps deriving clauses with more than one variable
    def is_domain_reduction(step):
        for c in step['derived']:
            if isinstance(c, cp.BoolVal): return True
            if isinstance(c, Operator) and c.name == "or":
                if len(c.args) > 1:
                    return False
            elif isinstance(c, Comparison):
                var, val = c.args
                assert isinstance(var, _NumVarImpl)
                assert is_num(val)
            else:
                raise ValueError(f"Unexpected derived constraint {c}")
        return True

        return len(get_variables(step['derived'])) <= 1

    proof = simplify_proof(proof, condition=is_domain_reduction)
    if do_sanity_check: sanity_check_proof(proof)
    if verbose > 0:
        print_proof_statistics(proof, "proof with only domain reductions")

    # Do the second minimization phase
    time_limit -= (time() - start)
    proof = minimize_proof(proof, model,
                           minimization_type=minimization_phase2,
                           mus_type=mus_type, mus_solver=mus_solver,
                           verbose=verbose, time_limit=time_limit)
    if do_sanity_check: sanity_check_proof(proof)
    if verbose > 0:
        print_proof_statistics(proof, "proof after second minimization phase")

    # Convert proof to sequence, merge steps with same set of reasons
    sequence = finalize_sequence(proof)

    if verbose > 0:
        print_sequence_statistics(sequence)
    if verbose > 1:
        pretty_print_sequence(sequence)

    return sequence


from cpmpy.expressions.utils import flatlist
from cpmpy.expressions.core import Expression
def finalize_sequence(proof):
    """
    Finialize the sequence by
     - merging steps with equal reasons
     - materializing the reasons of each step
    """
    proof_dict = {step['id'] : step for step in proof}
    reasons_cache = dict()
    explanation = []
    for step in proof:
        reasons = frozenset(step['reasons'])
        if reasons in reasons_cache:
            prev_step = next(x for x in explanation if x['id'] == reasons_cache[reasons])
            prev_step['output_lits'].extend(step['derived'])

        else:
            expl_step = dict(
                id = step['id'],
                input_lits = minimize_literals(flatlist([proof_dict[id]['derived'] for id in step['reasons'] if isinstance(id,int)])),
                constraints = [cons for cons in step['reasons'] if isinstance(cons, Expression)],
                output_lits = list(step['derived'])
            )
            reasons_cache[reasons] = step['id']
            explanation.append(expl_step)

    return explanation
