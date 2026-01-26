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
from itertools import combinations

from tqdm import tqdm

GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"

MATCHGATE = "matchgate"

KAK = "kak"
BQSKit = "bqskit"

AGENT_GREEDY = "Greedy"
AGENT_RL = "RL"

import numpy as np

def _to_python_type(x):
    """Recursively convert numpy types to Python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    elif isinstance(x, (np.floating,)):
        return float(x)
    elif isinstance(x, (np.ndarray, list, tuple)):
        return [_to_python_type(i) for i in x]
    elif isinstance(x, dict):
        return {k: _to_python_type(v) for k, v in x.items()}
    else:
        return x
    
class CliffordTSynthesizer:
    def __init__(self, mode1, mode2, epsilon=None, recursion_degree=None,
                 KAK_SYNTHESIS_CACHE=None, U3_SYNTHESIS_CACHE=None, verbose=False):
        self.mode1 = mode1
        self.mode2 = mode2
        self.epsilon = epsilon
        self.recursion_degree = recursion_degree
        self.KAK_SYNTHESIS_CACHE = KAK_SYNTHESIS_CACHE if KAK_SYNTHESIS_CACHE is not None else {}
        self.U3_SYNTHESIS_CACHE = U3_SYNTHESIS_CACHE if U3_SYNTHESIS_CACHE is not None else {}
        self.verbose = verbose

    def __call__(self, circuit):
        from scripts.synthesis.global_synthesis import clifford_t_synthesis
        qc_synth, t_list = clifford_t_synthesis(
            circuit,
            MODE1=self.mode1,
            MODE2=self.mode2,
            verbose=self.verbose,
            KAK_SYNTHESIS_CACHE=self.KAK_SYNTHESIS_CACHE,
            U3_SYNTHESIS_CACHE=self.U3_SYNTHESIS_CACHE,
            epsilon=self.epsilon,
            recursion_depth=self.recursion_degree
        )
        return qc_synth, t_list


class CliffordTSynthesizerMatchgate:
    def __init__(self,  epsilon=None, recursion_degree=None,
                 KAK_SYNTHESIS_CACHE=None, U3_SYNTHESIS_CACHE=None, verbose=False):
        self.epsilon = epsilon
        self.recursion_degree = recursion_degree
        self.KAK_SYNTHESIS_CACHE = KAK_SYNTHESIS_CACHE if KAK_SYNTHESIS_CACHE is not None else {}
        self.U3_SYNTHESIS_CACHE = U3_SYNTHESIS_CACHE if U3_SYNTHESIS_CACHE is not None else {}
        self.verbose = verbose


    def __call__(self, circuit):
        from scripts.synthesis.global_synthesis import clifford_t_synthesis_matchgate
        qc_synth, t_list = clifford_t_synthesis_matchgate(
            circuit,
            verbose=self.verbose,
            U3_SYNTHESIS_CACHE=self.U3_SYNTHESIS_CACHE,
            epsilon=self.epsilon,
        )
        return qc_synth, t_list

class BaseOptimizer:
    def __init__(self,
                 synthesis_config,
                 model_config,
                 task_config = None,
                 target_circuit_and_name = (None, None),
                 KAK_SYNTHESIS_CACHE=None,
                 U3_SYNTHESIS_CACHE=None):
        """
        Initialize the optimizer using configuration dictionaries.

        Args:
            task_config (dict): Contains N_QUBITS, N_GATES, SEED, COUPLING_MAP.
            synthesis_config (dict): Contains EPS (epsilon value for synthesis).
            model_config (dict): Contains ONLY_INTERACTIONS flag.
        """
        from qiskit.circuit.random import random_circuit

        # TASK SETTING
        if task_config is None and target_circuit_and_name[0] is None:
            assert False, "Either task_config or target_circuit must be provided."
        if task_config is not None and target_circuit_and_name[0] is not None:
            assert False, "Both task_config and target_circuit cannot be provided."
        if task_config is not None:
            self.target_is_random = True
            self.task_config = task_config
            
            self.original_circuit = random_circuit(
                num_qubits=task_config["N_QUBITS"],
                depth=task_config["N_GATES"],
                max_operands=2,  # optionally make this configurable
                seed=task_config["SEED"]
            )
            self.only_interactions = model_config.get("ONLY_INTERACTIONS", True)
            coupling_map = task_config.get("COUPLING_MAP", None)
            if coupling_map is not None:
                from scripts.agents.utils import refine_from_coupling_map
                self.original_circuit = refine_from_coupling_map(self.original_circuit, coupling_map)
                print("Refined circuit based on coupling map.")

        if target_circuit_and_name[0] is not None:
            self.target_is_random = False
            self.target_circuit_name = target_circuit_and_name[1]
            self.original_circuit = target_circuit_and_name[0]
            print(f"Target circuit name: {self.target_circuit_name}")
            print(f"Number of qubits in target circuit: {self.original_circuit.num_qubits}")
            self.only_interactions = True


        self.synthesis_config = synthesis_config
        self.model_config = model_config


        self.MODE1 = synthesis_config["MODE1"]
        assert self.MODE1 in [GRIDSYNTH, SOLOVAY_KITAEV, MATCHGATE], "Invalid 1q synthesis mode."

        self.MODE2 = synthesis_config["MODE2"]
        assert self.MODE2 in [KAK, BQSKit, MATCHGATE], "Invalid 2q synthesis mode."

        self.recursion_degree = None
        self.epsilon = None

        if self.MODE1 == SOLOVAY_KITAEV:
            self.recursion_degree = synthesis_config.get("RECURSION_DEGREE", 2)
        if self.MODE1 == GRIDSYNTH or self.MODE1 == MATCHGATE:
            self.epsilon = synthesis_config.get("EPS", 0.01)

        self.initial_fidelity = None
        if KAK_SYNTHESIS_CACHE is None:
            KAK_SYNTHESIS_CACHE = {}
        if U3_SYNTHESIS_CACHE is None:
            U3_SYNTHESIS_CACHE = {}

        self.KAK_SYNTHESIS_CACHE = KAK_SYNTHESIS_CACHE
        self.U3_SYNTHESIS_CACHE = U3_SYNTHESIS_CACHE

        initial_time = time.time()
        self.initial_time = initial_time

        self._preprocess_circuit()
        self.applied_merges = []


        self.synthesizer = CliffordTSynthesizer(
            mode1=self.MODE1,
            mode2=self.MODE2,
            epsilon=self.epsilon,
            recursion_degree=self.recursion_degree,
            KAK_SYNTHESIS_CACHE=KAK_SYNTHESIS_CACHE,
            U3_SYNTHESIS_CACHE=U3_SYNTHESIS_CACHE,
            verbose=False
        )

        if self.MODE1 == MATCHGATE:
            self.synthesizer = CliffordTSynthesizerMatchgate(
                epsilon=self.epsilon,
                U3_SYNTHESIS_CACHE=U3_SYNTHESIS_CACHE,
                verbose=False
            )

        _, t_list = self._synthesize(self.base_circuit, first_run=True)
        self.initial_tcount = sum(t_list)

        final_time = time.time()
        self.print_initial_stats(time_taken=final_time - initial_time)


        self.best_tcount = None
        self.optimized_circuit = None
        self.logs = []
        self._prepare_log_directory()
        self._dump_configs()
        self._dump_initial_stats(time_taken=final_time - initial_time)



    def optimizer_name(self) -> str:
        """Each subclass must define its optimizer name (e.g. 'Greedy', 'RL')."""
        pass

    def optimize(self):
        """Run the optimization process."""
        # return sequence, tcount
        pass

    def conclude_optimization(self):
        """Finalize the optimization process and log results."""
        self.report()
        self._dump_final_stats()

    def _preprocess_circuit(self):
        """Clean up the circuit by removing barriers and identities."""
        self.base_circuit = remove_barriers(self.original_circuit)
        self.base_circuit = remove_identities(self.base_circuit)
        print("Preprocessing done. Circuit cleaned.")

    def _synthesize(self, circuit, first_run=False):
        starting_time = time.time()
        qc_synth, tcount = self.synthesizer(circuit)
        time_taken = time.time() - starting_time

        if first_run:
            print(f"First synthesis completed in {time_taken:.2f} seconds.")
            if self.base_circuit.num_qubits < 2:
                fidelity = compute_fidelity(circuit, qc_synth)
                self.initial_fidelity = fidelity
                print(f"Synthesis done. T-count: {tcount}, Fidelity: {self.initial_fidelity:.6f}")

        return qc_synth, tcount




    def print_initial_stats(self, time_taken):
        """Print basic statistics about the input circuit before optimization."""
        num_qubits = self.base_circuit.num_qubits
        num_gates = len(self.base_circuit.data)

        # All possible qubit pairs
        full_merge_set = set(combinations(range(num_qubits), 2))

        # Actual merge candidates
        possible_merges = get_possible_merges(self.base_circuit, only_interactions=self.only_interactions)
        merge_set = set(possible_merges)

        num_possible_merges = len(merge_set)
        num_all_merges = len(full_merge_set)

        print("========== Initial Circuit Statistics ==========")
        print(f"Number of qubits:       {num_qubits}")
        print(f"Number of gates:        {num_gates}")
        print(f"Initial T-count:        {self.initial_tcount}")
        print(f"Time taken:             {time_taken:.2f} seconds")
        print(f"Possible merges:        {num_possible_merges}")

        if merge_set == full_merge_set:
            print("Merge strategy:         All qubit pairs considered")
        else:
            print(f"Merge strategy:         {num_possible_merges} / {num_all_merges} qubit pairs selected (interacting only)")
        print("=================================================\n\n")


    def report(self):
        """Print a summary of the optimization."""
        print("========== Optimization Report ==========")
        print(f"Applied merges: {self.applied_merges}")
        print(f"Final T-count: {self.best_tcount}")
        self.final_fidelity = None
        if self.base_circuit.num_qubits < 10:
            self.synthesized_circuit, _ = self._synthesize(self.optimized_circuit)
            final_fidelity = compute_fidelity(self.original_circuit, self.synthesized_circuit)
            self.final_fidelity = final_fidelity
            print(f"Final Fidelity: {final_fidelity:.6f}")
        end_time = time.time()
        print(f"Time taken: {(end_time - self.initial_time)/60:.2f} minutes")
        print("=========================================")

    def get_optimized_circuit(self):
        """Return the optimized quantum circuit."""
        return self.optimized_circuit

    def _prepare_log_directory(self):
        """Set up folder structure for outputs."""
        base_dir = f"outputs/{self.optimizer_name()}"
        MODE = f"{self.synthesis_config['MODE1']}_x_{self.synthesis_config['MODE2']}"

        #print(MODE)
        if self.target_is_random == True:
            circuit_type = self.task_config.get("TYPE", "undefined")  # You’ll set this manually in main
            exp_name = f"GeneralRandomCircuit_{self.task_config['N_QUBITS']}_{self.task_config['N_GATES']}_{circuit_type}"
            seed = self.task_config.get("SEED", 0)
        elif self.target_is_random == False:
            exp_name = f"Target_{self.target_circuit_name}"
            seed = 0  # No seed for predefined circuits
        additional_info = None
        if self.MODE1 == GRIDSYNTH or self.MODE1 == MATCHGATE:
            additional_info = f"epsilon_{self.synthesis_config['EPS']:.5f}"
        elif self.MODE1 == SOLOVAY_KITAEV:
            additional_info = f"depth_{self.synthesis_config['DEPTH']}"
        self.exp_path = os.path.join(base_dir,
                                     MODE,
                                     exp_name,
                                     f"seed_{seed}",
                                     additional_info)

        os.makedirs(os.path.join(self.exp_path, "configs"), exist_ok=True)

    def _dump_configs(self):
        """Save task, synthesis, and model configs."""
        if self.target_is_random == True:
            with open(os.path.join(self.exp_path, "configs", "task_config.yaml"), "w") as f:
                yaml.dump(self.task_config, f)

        with open(os.path.join(self.exp_path, "configs", "synthesis_config.yaml"), "w") as f:
            yaml.dump(self.synthesis_config, f)

        with open(os.path.join(self.exp_path, "configs", "model_config.yaml"), "w") as f:
            yaml.dump(self.model_config, f)

    def _dump_initial_stats(self, time_taken):
        """Save initial stats to file."""
        num_qubits = self.base_circuit.num_qubits
        num_gates = len(self.base_circuit.data)

        full_merge_set = set(combinations(range(num_qubits), 2))
        possible_merges = get_possible_merges(self.base_circuit, only_interactions=self.only_interactions)
        merge_set = set(possible_merges)

        num_possible_merges = len(merge_set)
        num_all_merges = len(full_merge_set)

        stats = {
            "Number of qubits": int(num_qubits),
            "Number of gates": int(num_gates),
            "Initial T-count": int(self.initial_tcount),
            "Initial Fidelity": float(self.initial_fidelity) if self.initial_fidelity is not None else None,
            "Time taken (m)": float(round(time_taken/60, 2)),
            "Possible merges": int(num_possible_merges),
            "Total possible pairs": int(num_all_merges),
            "Merge strategy": (
                "All qubit pairs"
                if merge_set == full_merge_set
                else f"{num_possible_merges} / {num_all_merges} pairs selected"
            )
        }

        with open(os.path.join(self.exp_path, "initial_stats.yaml"), "w") as f:
            yaml.dump(stats, f)

    def _dump_final_stats(self):
        """Save final stats to file."""
        end_time = time.time()
        final_stats = {
            "Final T-count": self.best_tcount,
            "Final Fidelity": self.final_fidelity if self.final_fidelity is not None else None,
            "Number of merges": len(self.applied_merges),
            "Applied merges": self.applied_merges,
            "Duration (min)": round((end_time - self.initial_time) / 60, 2)
        }

        # sanitize numpy types
        final_stats = _to_python_type(final_stats)

        with open(os.path.join(self.exp_path, "final_stats.yaml"), "w") as f:
            yaml.dump(final_stats, f)

    def _log_step(self, temp_merge, t_count, eval_time_sec, total_time_sec):
        step_id = len(self.logs) + 1
        self.logs.append({
            "step_id": step_id,
            "temp_merge": temp_merge,
            "applied_merges": deepcopy(self.applied_merges),
            "best_t_count": self.best_tcount,
            "t_count": t_count,
            "eval_time_sec": eval_time_sec,
            "total_time_sec": total_time_sec,
        })

