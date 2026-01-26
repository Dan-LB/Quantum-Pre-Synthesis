import os
import json
import time
from copy import deepcopy
from itertools import permutations

from tqdm import tqdm

from scripts.agents.base_agent import BaseOptimizer
from scripts.agents.utils import get_possible_merges
from scripts.mergers import apply_sequence_merge


MATCHGATE = "matchgate"
GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"
KAK = "kak"
BQSKit = "bqskit"


def commutes(op1, op2):
    """
    Two merge operations commute if they share no common qubits/elements.
    op1, op2: tuples, e.g. (0, 1) and (2, 3)
    """
    return set(op1).isdisjoint(set(op2))

def get_canonical_plan(plan):
    """
    Returns the lexicographically minimal (canonical) version of the plan
    by sorting commuting operations.
    """
    # Work on a list copy to avoid side effects
    p = list(plan)
    n = len(p)
    
    # Bubble sort approach: 
    # Swap adjacent elements if they are out of order AND they commute.
    changed = True
    while changed:
        changed = False
        for i in range(n - 1):
            # If left is lexically larger than right...
            if p[i] > p[i+1]:
                # ...and they share no elements...
                if commutes(p[i], p[i+1]):
                    # ...swap to canonicalize.
                    p[i], p[i+1] = p[i+1], p[i]
                    changed = True
    return p

class FullSearchMergeOptimizer(BaseOptimizer):
    """
    Exhaustive search over all possible plans (permutations of merge subsets)
    without repetition. Depth = total number of merges.

    For possible merges M, this tests all:
        permutations(M, 1) +
        permutations(M, 2) +
        ...
        permutations(M, len(M))

    Example (4-qubit brickwork):
        possible_merges = [(0,1), (1,2), (2,3), (3,0)]
        total plans = 64.

    This optimizer replaces the greedy optimizer entirely.
    """

    def optimizer_name(self):
        return "FullSearch"

    # ----------------------------------------------------------------------
    # Generate all plans (permutations of subsets without repetition)
    # ----------------------------------------------------------------------
    def _generate_all_plans(self, merges):
        """
        Enumerate all non-empty permutations of all subsets of merges.

        merges: list of merge operations such as [(0,1),(1,2),(2,3),(3,0)]

        Yields:
            plan (list of merges)
        """
        n = len(merges)
        for r in range(1, n + 1):
            for plan in permutations(merges, r):
                yield list(plan)

    # ----------------------------------------------------------------------
    # Evaluate a single plan
    # ----------------------------------------------------------------------
    def _evaluate_plan(self, plan, refine=False):
        """
        Apply the sequence of merges, synthesize, return T-count.

        plan: list of merges

        Returns:
            tcount (int)
        """

        if plan != get_canonical_plan(plan):
            # Non-canonical plan, skip evaluation
            return float("inf")

        temp = apply_sequence_merge(
            deepcopy(self.base_circuit),
            plan,
            refine=refine
        )

        try:
            _, tcounts = self._synthesize(temp)
            return sum(tcounts)
        except Exception:
            # Invalid synthesis or merge, return sentinel value
            return float("inf")

    # ----------------------------------------------------------------------
    # Main full search routine
    # ----------------------------------------------------------------------
    def optimize(self, verbose=False):
        """
        Run exhaustive search over all plans.

        Returns:
            best_plan (list of merges)
            best_tcount (int)
        """
        start_time = time.time()

        # Determine possible merges
        possible_merges = get_possible_merges(
            self.base_circuit,
            only_interactions=self.only_interactions
        )

        # Initial T-count for baseline
        _, init = self._synthesize(deepcopy(self.base_circuit))
        self.best_tcount = sum(init)
        self.best_plan = []
        if verbose:
            print(f"Initial T-count: {self.best_tcount}")

        # Prepare logging
        self.log_file = open(os.path.join(self.exp_path, "log.json"), "w")

        # Refine if matchgate-like synthesis is used
        refine = (self.MODE1 == MATCHGATE) or (self.MODE2 == MATCHGATE)

        # Precompute number of total plans
        # Total = sum_{r=1..n} P(n,r)
        n = len(possible_merges)
        total_plans = 0
        for r in range(1, n + 1):
            # P(n,r) = n! / (n-r)!
            # compute manually to avoid import math.perm if Python old
            num = 1
            for i in range(n - r + 1, n + 1):
                num *= i
            total_plans += num

        if verbose:
            print(f"Total plans to evaluate: {total_plans}")

        pbar = tqdm(total=total_plans, desc="FullSearch", unit="plan")

        # ------------------------------------------------------------------
        # Enumerate all plans
        # ------------------------------------------------------------------
        for plan in self._generate_all_plans(possible_merges):
            t0 = time.time()

            tcount = self._evaluate_plan(plan, refine=refine)

            # Update best plan
            if tcount < self.best_tcount:
                self.best_tcount = tcount
                self.best_plan = plan
                if verbose:
                    print(f"New best T-count {tcount} with plan {plan}")

            t1 = time.time()

            # Log each plan evaluation
            self._log_step(
                temp_merge=plan[-1],
                t_count=tcount,
                eval_time_sec=t1 - t0,
                total_time_sec=t1 - start_time
            )

            pbar.update(1)

        pbar.close()
        self.log_file.close()

        # Produce final optimized circuit
        self.optimized_circuit = apply_sequence_merge(
            deepcopy(self.base_circuit),
            self.best_plan,
            refine=refine
        )

        self.conclude_optimization()

        return self.best_plan, self.best_tcount
