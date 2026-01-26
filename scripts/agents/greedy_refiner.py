import time
from copy import deepcopy
from tqdm import tqdm
from scripts.agents.base_agent import BaseOptimizer
from scripts.mergers import apply_sequence_merge

class GreedyRefiner(BaseOptimizer):
    """
    Progressive greedy refinement optimizer.
    For each prefix of an initial merge sequence, performs greedy optimization starting from that prefix.
    """

    def __init__(self, synthesis_config, model_config, task_config=None,
                 target_circuit_and_name=(None, None),
                 initial_sequence=None,
                 KAK_SYNTHESIS_CACHE=None,
                 U3_SYNTHESIS_CACHE=None):
        super().__init__(synthesis_config,
                         model_config,
                         task_config,
                         target_circuit_and_name,
                         KAK_SYNTHESIS_CACHE,
                         U3_SYNTHESIS_CACHE)

        # Store a clean copy of the original base circuit
        self.original_base_circuit = deepcopy(self.base_circuit)

        # Normalize all initial merges to tuples
        self.initial_sequence = [tuple(m) for m in (initial_sequence)]

    def optimizer_name(self):
        return "RL_refined"

    def optimize(self, verbose=True):
        """
        Perform progressive greedy refinement from prefixes of the initial sequence.
        Returns:
            results: dict mapping prefix length -> {"prefix", "sequence", "tcount", "duration_sec"}
        """
        results = {}
        total_stages = len(self.initial_sequence) + 1
        global_start = time.time()

        if verbose:
            print(f"Starting GreedyRefiner with {total_stages} stages...")

        BEST_SEQUENCE = []
        BEST_TCOUNT = float("inf")

        for k in range(total_stages):
            prefix = self.initial_sequence[:k]

            if verbose:
                print(f"\n=== Stage {k+1}/{total_stages}: prefix = {prefix} ===")

            # Initialize best_tcount safely from the stage circuit
            stage_circuit = apply_sequence_merge(deepcopy(self.original_base_circuit), prefix)
            synth_result = self._synthesize(stage_circuit)
            stage_best_tcount = sum(synth_result[1]) if synth_result[1] else float("inf")

            # Run greedy optimization starting from this prefix on a fresh circuit
            seq, tcount = self._greedy_optimize(
                deepcopy(self.original_base_circuit),
                prefix.copy(),
                stage_best_tcount,
                verbose=verbose
            )



            # Track global best sequence
            if tcount < BEST_TCOUNT:
                BEST_SEQUENCE = seq
                BEST_TCOUNT = tcount

        # Set final optimizer state only once
        self.applied_merges = BEST_SEQUENCE
        self.best_tcount = BEST_TCOUNT
        self.optimized_circuit = apply_sequence_merge(deepcopy(self.base_circuit), self.applied_merges)
        self.conclude_optimization()

        total_time = time.time() - global_start
        if verbose:
            print(f"\nGreedy refinement complete. Total duration: {total_time:.2f} s")

        return BEST_SEQUENCE, BEST_TCOUNT


    def _greedy_optimize(self, base_circuit, applied_merges, best_tcount, verbose=True):
        """
        Internal greedy merge logic (adapted from GreedyMergeOptimizer).
        Ensures merges are tuples and T-count is minimized.
        """
        from scripts.agents.utils import get_possible_merges

        possible_merges = get_possible_merges(base_circuit, only_interactions=self.only_interactions)
        # Filter out already applied merges
        possible_merges = [tuple(m) for m in possible_merges if tuple(m) not in applied_merges]

        pbar = tqdm(total=len(possible_merges), desc="Optimizing", unit="eval", leave=False)

        while possible_merges:
            best_candidate = None
            best_candidate_tcount = float("inf")

            for merge in possible_merges:
                temp_merges = applied_merges + [merge]

                refine = True
                if getattr(self, "MODE1", None) == "matchgate" or getattr(self, "MODE2", None) == "matchgate":
                    refine = False

                temp_circuit = apply_sequence_merge(deepcopy(base_circuit), temp_merges, refine=refine)

                try:
                    _, tcount_list = self._synthesize(temp_circuit)
                    tcount = sum(tcount_list) if tcount_list else float('inf')
                except Exception as e:
                    if verbose:
                        print(f"Failed synthesis on {merge}: {e}")
                    pbar.update(1)
                    continue

                if tcount < best_candidate_tcount:
                    best_candidate = merge
                    best_candidate_tcount = tcount

                pbar.update(1)

            if best_candidate is None or best_candidate_tcount >= best_tcount:
                if verbose:
                    print("No improving merge found. Terminating stage.")
                break

            applied_merges.append(best_candidate)
            possible_merges.remove(best_candidate)
            best_tcount = best_candidate_tcount

            if verbose:
                print(f"Added merge {best_candidate}, new T-count: {best_tcount}")

        pbar.close()

        # Return final stage results
        return applied_merges, best_tcount
