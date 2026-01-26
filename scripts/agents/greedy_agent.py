import os
import yaml
import json
import datetime


import time
from copy import deepcopy
from scripts.synthesis.global_synthesis import clifford_t_synthesis
from scripts.agents.utils import get_possible_merges

from scripts.mergers import apply_sequence_merge, remove_barriers, remove_identities

from scripts.utils import compute_fidelity
from scripts.agents.base_agent import BaseOptimizer

from tqdm import tqdm

GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"

MATCHGATE = "matchgate"

KAK = "kak"
BQSKit = "bqskit"

class GreedyMergeOptimizer(BaseOptimizer):
    def __init__(self,
                 synthesis_config,
                 model_config,
                 task_config=None,
                 target_circuit_and_name=(None, None),
                 KAK_SYNTHESIS_CACHE=None,
                 U3_SYNTHESIS_CACHE=None):
        super().__init__(synthesis_config,
                         model_config,
                         task_config,
                         target_circuit_and_name,
                         KAK_SYNTHESIS_CACHE,
                         U3_SYNTHESIS_CACHE)

    def optimizer_name(self):
        return "Greedy"


    def optimize(self, verbose=False):
        """
        Run greedy merge optimization using tqdm for progress tracking.

        Returns:
            (List[Tuple[int, int]], int): Sequence of applied merges, and final T-count.
        """
        start_time = time.time()
        possible_merges = get_possible_merges(self.base_circuit, only_interactions=self.only_interactions)

        _, self.best_tcount = self._synthesize(self.base_circuit)
        self.best_tcount = sum(self.best_tcount)

        if verbose:
            print(f"Initial T-count: {self.best_tcount}")

        # Estimate max steps for tqdm
        total_evaluations = (len(possible_merges) * (len(possible_merges) + 1)) // 2
        pbar = tqdm(total=total_evaluations, desc="Optimizing", unit="eval")
        self.log_file = open(os.path.join(self.exp_path, "log.json"), "w")


        while possible_merges:
            best_candidate = None
            best_candidate_tcount = float("inf")

            for merge in possible_merges:
                temp_merges = self.applied_merges + [merge]
                refine = False
                if self.MODE1 == MATCHGATE or self.MODE2 == MATCHGATE:
                    refine = True

                temp_circuit = apply_sequence_merge(deepcopy(self.base_circuit), temp_merges, refine=refine)
                #print(f"Printing temp circuit for merge {merge} in {temp_merges}:")
                #print(temp_circuit)
                #print("-----\n")

                eval_start = time.time()
                try:
                    _, tcount = self._synthesize(temp_circuit)
                    tcount = sum(tcount)
                except Exception as e:
                    if verbose:
                        print(f"❌ Failed synthesis on {merge}: {e}")
                    pbar.update(1)
                    continue
                eval_end = time.time()

                # --- Debugging output for each evaluated merge ---
                if verbose:
                    delta = tcount - self.best_tcount
                    if delta < 0:
                        print(f"✅ Merge {merge} IMPROVES T-count by {-delta} (new: {tcount}, old: {self.best_tcount})")
                    elif delta == 0:
                        print(f"⚪ Merge {merge} yields SAME T-count ({tcount})")
                    else:
                        print(f"❌ Merge {merge} worsens T-count by {delta} (new: {tcount}, old: {self.best_tcount})")

                # --- Best candidate update ---
                if tcount < best_candidate_tcount:
                    best_candidate = merge
                    best_candidate_tcount = tcount

                pbar.update(1)

                self._log_step(temp_merges, tcount, eval_end - eval_start, eval_end - start_time)

            if best_candidate is None or best_candidate_tcount >= self.best_tcount:
                if verbose:
                    print("No improving merge found. Terminating.")
                break

            self.applied_merges.append(best_candidate)
            possible_merges.remove(best_candidate)
            self.best_tcount = best_candidate_tcount

            if verbose:
                print(f"⭐ Added merge {best_candidate}, new best T-count: {self.best_tcount}")


        pbar.close()
        self.log_file.close()


        end_time = time.time()
        self.duration_sec = end_time - start_time
        self.duration_min = self.duration_sec / 60

        self.optimized_circuit = apply_sequence_merge(deepcopy(self.base_circuit), self.applied_merges)
        self.conclude_optimization()
        greedy_sequence, greedy_tcount = self.applied_merges, self.best_tcount
        return greedy_sequence, greedy_tcount

