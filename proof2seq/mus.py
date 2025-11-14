"""
    Stateful implementation of MUS-algorithms present in CPMPy
"""

import cpmpy as cp
from cpmpy.tools.explain import make_assump_model
from cpmpy.expressions.core import Expression

from .utils import get_variables


class MUSAlgo:

    def __init__(self, proof, constraints, mus_solver):
        """
        Initialize the MUS algorithm.
        :param proof:
        :param constraints:
        :param solver:
        """
        assump_model, self.cons, self.assump = make_assump_model(constraints)
        self.assump = list(self.assump)

        self.proof_dict = {step['id'] : step for step in proof}

        # now add the derived constraints in the proof too
        for step in proof:
            bv = cp.boolvar()
            self.assump.append(bv)
            self.cons.append(step['id'])
            assump_model += bv.implies(cp.all(step['derived']))

        self.dmap = dict(zip(self.assump, self.cons))
        self.rev_map = dict(zip(self.cons, self.assump))

        self.solver = cp.SolverLookup.get(mus_solver, assump_model)

    def get_mus(self, soft, hard):
        raise NotImplementedError

    def get_assumps(self, lst):

        assumps = []
        for x in lst:
            if x not in self.rev_map:
                assert isinstance(x, Expression)
                bv = cp.boolvar()
                self.solver += bv.implies(x)
                self.rev_map[x] = bv
                self.dmap[bv] = x
                assumps.append(bv)
            else:
                assumps.append(self.rev_map[x])
        return assumps

class DeletionBasedMUS(MUSAlgo):

    def get_mus(self, soft, hard):

        soft_assump = self.get_assumps(soft)
        hard_assump = self.get_assumps(hard)

        assert self.solver.solve(assumptions=soft_assump+hard_assump) is False

        core = set(self.solver.get_core()) - set(hard_assump)
        for c in sorted(core, key=lambda c: -len(get_variables(self.dmap[c]))):
            if c not in core:
                continue  # already removed
            core.remove(c)
            if self.solver.solve(assumptions=list(core) + hard_assump) is True:
                core.add(c)
            else:  # UNSAT, use new solver core (clause set refinement)
                core = set(self.solver.get_core()) - set(hard_assump)

        return [self.dmap[a] for a in core]

class SMUS(MUSAlgo):

    def __init__(self, proof, constraints, mus_solver, hs_solver="gurobi"):
        super().__init__(proof, constraints, mus_solver)

        self.hs_solver = hs_solver

    def _get_val(self, item):
        if isinstance(item, Expression):
            return item.value()
        if isinstance(item, int):
            item = self.proof_dict[item]
        if isinstance(item, dict) and "derived" in item:
            return all(expr.value() for expr in item["derived"])
        raise ValueError(f"Cannot compute value of given item, expected `Expression` or proof step, but got {item}")


    def get_mus(self, soft, hard):

        if len(soft) == 0:
            return soft

        soft_assump = self.get_assumps(soft)
        hard_assump = self.get_assumps(hard)

        hs_solver = cp.SolverLookup.get(self.hs_solver)
        hs_solver.minimize(cp.sum(soft_assump))

        while hs_solver.solve() is True:

            hs = [a for a in soft_assump if a.value()]

            if self.solver.solve(assumptions=hs+hard_assump) is False:
                # UNSAT, found MUS
                return [self.dmap[a] for a in hs]

            # else SAT, find some (cheap) correction subsets
            new_corr_subset = [a for a in soft_assump if a.value() is False and self._get_val(self.dmap[a]) is False]
            hs_solver += cp.sum(new_corr_subset) >= 1

            # greedily search for other corr subsets disjoint to this one
            sat_subset = list(new_corr_subset)
            while self.solver.solve(assumptions=sat_subset+hard_assump) is True:
                new_corr_subset = [a for a in soft_assump if a.value() is False and self._get_val(self.dmap[a]) is False]
                sat_subset += new_corr_subset
                hs_solver += cp.sum(new_corr_subset) >= 1

        raise ValueError("HS solver is UNSAT, this should not happen!")
















