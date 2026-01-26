from itertools import product
import sys
from typing import List, Tuple
from tqdm import tqdm
from qiskit import QuantumCircuit
import numpy as np

from scripts.agents.RL_agent import RLMergeOptimizer
from scripts.agents.greedy_agent import GreedyMergeOptimizer
from scripts.agents.greedy_refiner  import GreedyRefiner

GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"

KAK = "kak"
BQSKit = "bqskit"

N_QUBITSS = [10]
TYPES = ["Full"]
SEEDS = [0, 1, 2, 3, 4]
Combinations = [(SOLOVAY_KITAEV, KAK), (GRIDSYNTH, BQSKit), (GRIDSYNTH, KAK)]


def create_random_brick(n_qubits, n_layers, seed):
    np.random.seed(seed)
    QC = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for qubit in range(n_qubits):
            QC.ry(np.random.uniform(0, 2 * np.pi), qubit)
            QC.rz(np.random.uniform(0, 2 * np.pi), qubit)
        for qubit in range(int(n_qubits / 2)):
            QC.rxx(np.random.uniform(0, 2 * np.pi), 2*qubit, 2*qubit + 1)
        for qubit in range(int(n_qubits / 2)):
            QC.rxx(np.random.uniform(0, 2 * np.pi), 2*qubit+1, (2*qubit + 2)%n_qubits)
    return QC

def create_random_controlledRx(n_qubits, n_layers, seed):
    np.random.seed(seed)
    QC = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for qubit in range(n_qubits):
            QC.ry(np.random.uniform(0, 2 * np.pi), qubit)
            QC.rz(np.random.uniform(0, 2 * np.pi), qubit)
        for qubit in range(n_qubits - 1):
            QC.crx(np.random.uniform(0, 2 * np.pi), qubit, qubit + 1)
        QC.crx(np.random.uniform(0, 2 * np.pi), n_qubits - 1, 0)  # Connect last qubit to first for circular entanglement
    return QC



experiment_list = []
for N_QUBITS in N_QUBITSS:
    N_GATES = N_QUBITS 
    for TYPE, SEED, (MODE1, MODE2) in product(TYPES, SEEDS, Combinations):
        experiment_list.append((N_QUBITS, N_GATES, TYPE, SEED, MODE1, MODE2))

print(f"Total experiments: {len(experiment_list)}")


for N_QUBITS, N_GATES, TYPE, SEED, MODE1, MODE2 in tqdm(experiment_list, desc="Running experiments for bricks (linear connectivity)"):


    circuit = create_random_brick(N_QUBITS, 3, seed=SEED)
    circuit_name = f"Brick_{N_QUBITS}_{SEED}"


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

    greedy_agent = GreedyMergeOptimizer(synthesis_config, model_config, target_circuit_and_name=(circuit, circuit_name))
    greedy_sequence, greedy_tcount = greedy_agent.optimize(verbose=False)
    greedy_agent.report()

    RL_agent = RLMergeOptimizer(synthesis_config, model_config, target_circuit_and_name=(circuit, circuit_name))
    rl_sequence, rl_tcount = RL_agent.optimize(verbose=False)
    RL_agent.report()

    refiner_agent = GreedyRefiner(synthesis_config, model_config, target_circuit_and_name=(circuit, circuit_name), initial_sequence=rl_sequence)
    refined_sequence, refined_tcount = refiner_agent.optimize(verbose=False)
    refiner_agent.report()




for N_QUBITS, N_GATES, TYPE, SEED, MODE1, MODE2 in tqdm(experiment_list, desc="Running experiments for controlled Rx (linear connectivity)"):


    circuit = create_random_controlledRx(N_QUBITS, 3, seed=SEED)
    circuit_name = f"ControlledRx_{N_QUBITS}_{SEED}"

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

    greedy_agent = GreedyMergeOptimizer(synthesis_config, model_config, target_circuit_and_name=(circuit, circuit_name))
    greedy_sequence, greedy_tcount = greedy_agent.optimize(verbose=False)
    greedy_agent.report()

    RL_agent = RLMergeOptimizer(synthesis_config, model_config, target_circuit_and_name=(circuit, circuit_name))
    rl_sequence, rl_tcount = RL_agent.optimize(verbose=False)
    RL_agent.report()

    refiner_agent = GreedyRefiner(synthesis_config, model_config, target_circuit_and_name=(circuit, circuit_name), initial_sequence=rl_sequence)
    refined_sequence, refined_tcount = refiner_agent.optimize(verbose=False)
    refiner_agent.report()
