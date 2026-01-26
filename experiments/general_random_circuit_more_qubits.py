from itertools import product
import sys
from typing import List, Tuple
from tqdm import tqdm

from scripts.agents.RL_agent import RLMergeOptimizer
from scripts.agents.greedy_agent import GreedyMergeOptimizer
from scripts.agents.greedy_refiner  import GreedyRefiner

GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"

KAK = "kak"
BQSKit = "bqskit"

N_QUBITSS = range(10, 26, 3)
TYPES = ["Full"]
SEEDS = [0, 1, 2, 3, 4]
Combinations = [(SOLOVAY_KITAEV, KAK), (GRIDSYNTH, BQSKit), (GRIDSYNTH, KAK)]





experiment_list = []
for N_QUBITS in N_QUBITSS:
    N_GATES = N_QUBITS 
    for TYPE, SEED, (MODE1, MODE2) in product(TYPES, SEEDS, Combinations):
        experiment_list.append((N_QUBITS, N_GATES, TYPE, SEED, MODE1, MODE2))

print(f"Total experiments: {len(experiment_list)}")


for N_QUBITS, N_GATES, TYPE, SEED, MODE1, MODE2 in tqdm(experiment_list, desc="Running experiments for general random circuits"):


    couple_mapping = None

    task_config = {
        "N_QUBITS": N_QUBITS,
        "N_GATES": N_GATES,
        "SEED": SEED,
        "COUPLING_MAP": couple_mapping,
        "TYPE": TYPE
    }

    synthesis_config = {
        "EPS": 0.01,
        "DEPTH": 2,  # for solovay_kitaev if used
        "MODE1": MODE1,
        "MODE2": MODE2
    }

    model_config = {
        "ONLY_INTERACTIONS": True,
    }

    greedy_agent = GreedyMergeOptimizer(synthesis_config, model_config, task_config=task_config)
    greedy_sequence, greedy_tcount = greedy_agent.optimize(verbose=False)
    greedy_agent.report()

