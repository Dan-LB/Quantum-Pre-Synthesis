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

N_QUBITSS = [4, 6, 8, 10]#, 8, 10]
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
        "AGENT_CODE": "PPO",
        "TOTAL_TIMESTEPS": 10_000_000, #10_000_000
        "INITIAL_STEPS": 2048, #2048
        "BATCH_SIZE": 1024, #1024
        "ENTROPY_COEFFICIENT": 0.05,
        "EVAL_BEST_OF": 25,
        "ONLY_INTERACTIONS": True,
    }

    greedy_agent = GreedyMergeOptimizer(synthesis_config, model_config, task_config=task_config)
    greedy_sequence, greedy_tcount = greedy_agent.optimize(verbose=False)
    greedy_agent.report()

    RL_agent = RLMergeOptimizer(synthesis_config, model_config, task_config=task_config)
    rl_sequence, rl_tcount = RL_agent.optimize(verbose=False)
    RL_agent.report()

    refiner_agent = GreedyRefiner(synthesis_config, model_config, task_config=task_config, initial_sequence=rl_sequence)
    refined_sequence, refined_tcount = refiner_agent.optimize(verbose=False)
    refiner_agent.report()
