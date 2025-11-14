import copy

import cpmpy as cp
from cpmpy.expressions.core import Expression
from cpmpy.transformations.normalize import toplevel_list

from .mus import DeletionBasedMUS, SMUS


def minimize_proof(proof, model, minimization_type="global", mus_type="smus", **mus_algo_kwargs):

    assert minimization_type in {"proof", "trim", "local", "global"}

    if minimization_type == "proof":
        return proof # do nothing

    if minimization_type == "trim":
        return trim_proof(proof)

    if mus_type == "mus":
        mus_algo = DeletionBasedMUS(proof, model.constraints, **mus_algo_kwargs)
    elif mus_type == "smus":
        mus_algo = SMUS(proof, model.constraints, **mus_algo_kwargs)
    else:
        raise ValueError(f"Unknown MUS type {mus_type}, expected 'mus' or 'smus'")


    required = {proof[-1]['id']}
    new_proof = []

    for step in reversed(proof):
        if step['id'] in required:
            # Step is required, let's minimize the reasons of this step
            # We want a MUS that minimizes the number of constraints
            # Additionally, we prefer nogoods that we need to explain already anyway
            # Hence, we have a 3-step approach to accomplish this, using 3 MUS calls

            if minimization_type == "local":
                candidate_nogoods = [r for r in step['reasons'] if isinstance(r, int)]
                candidate_cons = [r for r in step['reasons'] if isinstance(r, Expression)]
                assert len(candidate_nogoods) + len(candidate_cons) == len(step['reasons']), "Euhm something went wrong here..."

            if minimization_type == "global":
                candidate_nogoods = [x['id'] for x in proof if x['type'] == 'nogood' and x['id'] < step['id']]
                candidate_cons = toplevel_list(model.constraints, merge_and=False)

            # Minimize number of constraints
            required_cons = mus_algo.get_mus(soft=candidate_cons,
                                             hard=candidate_nogoods + [~cp.all(step['derived'])])

            # Minimize number of nogoods
            potential_known_nogoods = [r for r in candidate_nogoods if r in required]
            potential_new_nogoods = [r for r in candidate_nogoods if r not in required]

            required_new_nogoods = mus_algo.get_mus(soft=potential_new_nogoods,
                                                    hard=potential_known_nogoods + required_cons + [~cp.all(step['derived'])])

            required_known_nogoods = mus_algo.get_mus(soft=potential_known_nogoods,
                                                      hard=required_new_nogoods + required_cons + [~cp.all(step['derived'])])

            required_nogoods = required_new_nogoods + required_known_nogoods
            assert all(isinstance(ng, int) for ng in required_nogoods)

            # Update set of required steps
            required.update(required_nogoods)

            step = copy.deepcopy(step)
            step['reasons'] = required_cons + required_nogoods
            new_proof.insert(0, step)

    return new_proof





def trim_proof(proof):
    # work from back to front
    required = {proof[-1]['id']}
    new_proof = []
    for step in reversed(proof):
        if step['id'] in required:
            new_proof.insert(0, step)
            required.update(step['reasons'])
    return new_proof
