import gymnasium as gym
import numpy as np
from copy import deepcopy
from qiskit import QuantumCircuit
from typing import List, Tuple
from time import time
from collections import OrderedDict

from scripts.mergers import remove_barriers, remove_identities, merge_contiguous_2q_and_refine, apply_sequence_merge
from scripts.utils import fidelity_to_error
from scripts.agents.utils import refine_from_coupling_map
from scripts.synthesis.global_synthesis import clifford_t_synthesis


def merge_matrix(n: int, i: int, j: int) -> np.ndarray:
    i, j = (j, i) if i > j else (i, j)
    M = np.eye(n)
    M[i, j] = 1
    M[j, i] = 1
    return M


def get_plan_matrix(n: int, plan: List[Tuple[int, int]]) -> np.ndarray:
    M = np.eye(n)
    for merge in plan:
        i, j = merge
        M = merge_matrix(n, i, j) @ M
    return M


def get_two_qubit_matrix(n: int, last_synth_qc, applied_merges) -> np.ndarray:
    twoq_matrix = np.zeros((n, n), dtype=int)
    for instr, qargs, _ in last_synth_qc.data:
        if len(qargs) == 2:
            i = last_synth_qc.find_bit(qargs[0]).index
            j = last_synth_qc.find_bit(qargs[1]).index
            twoq_matrix[i, j] += 1
            twoq_matrix[j, i] += 1
    for i, j in applied_merges:
        twoq_matrix[i, j] = 0
        twoq_matrix[j, i] = 0
    return twoq_matrix


class QuantumMergeEnv(gym.Env):
    def __init__(self, qc, Synthesizer, max_steps=100, cache_size=100_000):
        assert isinstance(qc, QuantumCircuit), "qc must be a QuantumCircuit"

        qc = remove_barriers(qc)
        qc = remove_identities(qc)
        self.target_qc = deepcopy(qc)
        self.current_qc = deepcopy(qc)
        self.total_reward = 0.0
        self.info = []
        self.qubit_t_counts = {}

        self.used_pairs = self.get_used_pairs()
        n_qubits = self.current_qc.num_qubits
        self.n_qubits = n_qubits
        self.max_steps = max_steps

        self.action_space = gym.spaces.Discrete(len(self.used_pairs))
        self.Synthesizer = Synthesizer

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, n_qubits, n_qubits), dtype=np.float32
        )

        ### MEMOIZATION BLOCK ###
        self.cache_size = cache_size
        self.cache = OrderedDict()  # (key) plan tuple → (qc, synth, t_list)
        ##########################

        init_time = time()
        self.reset()
        final_time = time()
        self.print_initial_stats(final_time - init_time)

    def get_used_pairs(self):
        used_pairs_set = set()
        for instr, qargs, _ in self.target_qc.data:
            if len(qargs) == 2:
                i = self.target_qc.find_bit(qargs[0]).index
                j = self.target_qc.find_bit(qargs[1]).index
                if i > j:
                    i, j = j, i
                used_pairs_set.add((i, j))
        used_pairs = sorted(list(used_pairs_set))
        print(f"Used pairs: {used_pairs}")
        return used_pairs

    def decode_action(self, action):
        if action < 0 or action >= len(self.used_pairs):
            raise ValueError("Invalid action index.")
        return self.used_pairs[action]

    def reset(self, seed=None):
        self.current_qc = self.target_qc.copy()
        self.total_reward = 0.0
        self.applied_merges = []
        self.last_synth_qc, self.last_t_list = self.Synthesizer(self.current_qc)
        obs = self._get_obs()
        self.t_count = sum(self.t_counts[i, j] for i in range(self.n_qubits) for j in range(self.n_qubits))
        self.steps = 0
        return obs, {}

    def print_initial_stats(self, time_taken):
        num_qubits = self.target_qc.num_qubits
        num_gates = len(self.target_qc.data)
        from itertools import combinations
        full_merge_set = set(combinations(range(num_qubits), 2))
        merge_set = set(self.used_pairs)
        print("========== Initial Circuit Statistics ==========")
        print(f"Number of qubits:       {num_qubits}")
        print(f"Number of gates:        {num_gates}")
        print(f"Initial T-count:        {self.t_count}")
        print(f"Time taken:             {time_taken:.2f} seconds")
        print(f"Possible merges:        {len(merge_set)} / {len(full_merge_set)} qubit pairs selected")

    def _get_obs(self):
        diag_t_matrix = np.zeros((self.n_qubits, self.n_qubits))
        for q in range(self.n_qubits):
            diag_t_matrix[q, q] = self.last_t_list[q]
        self.t_counts = diag_t_matrix.copy()

        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                diag_t_matrix[i, j] += self.last_t_list[i] + self.last_t_list[j]

        twoq_matrix = get_two_qubit_matrix(self.n_qubits, self.last_synth_qc, self.applied_merges)
        plan_matrix = get_plan_matrix(self.n_qubits, self.applied_merges)
        self.t_count = sum(self.t_counts[i, j] for i in range(self.n_qubits) for j in range(self.n_qubits))

        diag_t_matrix = diag_t_matrix / (np.max(diag_t_matrix) + 1e-10)
        twoq_matrix = twoq_matrix / (np.max(twoq_matrix) + 1e-10)
        plan_matrix = plan_matrix / (np.max(plan_matrix) + 1e-10)

        return np.stack(
            [diag_t_matrix.astype(np.float32),
             twoq_matrix.astype(np.float32),
             plan_matrix.astype(np.float32)],
            axis=0,
        )

    ### MEMOIZATION BLOCK ###
    def _cache_get(self, key):
        """Retrieve a cached synthesis if available."""
        if key in self.cache:
            synth, t_list = self.cache[key]
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            return deepcopy(synth), deepcopy(t_list)
        return None

    def _cache_put(self, key, synth, t_list):
        """Store a new synthesis result, pruning old entries if needed."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (deepcopy(synth), deepcopy(t_list))
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # remove least recently used
    ##########################

    def step(self, action):
        i, j = self.decode_action(action)
        if (i, j) in self.applied_merges or i == j:
            info = {'episode': {'r': self.total_reward, 'l': self.steps}}
            self.info.append(info['episode'])
            return self._get_obs(), 0, True, True, {}

        new_plan = tuple(self.applied_merges + [(i, j)])

        ### MEMOIZATION BLOCK ###
        cached = self._cache_get(new_plan)
        if cached is not None:
            self.last_synth_qc, self.last_t_list = cached
        else:
            self.current_qc = apply_sequence_merge(deepcopy(self.target_qc), new_plan)
            temp_synth, self.last_t_list = self.Synthesizer(self.current_qc)
            self.last_synth_qc = temp_synth.copy()
            self._cache_put(new_plan, self.last_synth_qc, self.last_t_list)
        ##########################

        self.applied_merges.append((i, j))
        old_t_count = self.t_count
        obs = self._get_obs()
        new_t_count = self.t_count
        reward = old_t_count - new_t_count
        self.total_reward += reward
        self.t_count = new_t_count
        self.steps += 1

        terminated = self.t_count == 0
        truncated = self.steps >= self.max_steps
        info = {}
        if truncated:
            info['episode'] = {'r': self.total_reward, 'l': self.steps}
            self.info.append(info['episode'])

        return obs, reward, terminated, truncated, info
