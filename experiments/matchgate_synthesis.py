import os
import sys

from typing import Tuple, List, Optional
from qiskit import QuantumCircuit

script_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(script_dir)

KAK_SYNTHESIS_CACHE: dict[Tuple[Tuple[float, float], ...], QuantumCircuit] = {}
U3_SYNTHESIS_CACHE: dict[Tuple[float, float, float], Tuple[QuantumCircuit, List[str]]] = {}

GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"
MATCHGATE = "matchgate"


import numpy as np



def random_rz_rxx_circuit(n_qubits, n_gates, seed=0):
    """
    Build a random circuit composed of Rz and Rxx gates.
    """
    np.random.seed(seed)
    qc = QuantumCircuit(n_qubits)
    
    for _ in range(n_gates):
        gate_type = np.random.choice(["Rxx", "Rz"])
        if gate_type == "Rz":
            q = np.random.randint(0, n_qubits)
            alpha = np.random.uniform(0, 2*np.pi)
            qc.rz(alpha, q)
        elif gate_type == "Rxx":
            q1 = np.random.choice(range(n_qubits))
            q2 = q1 + 1 if q1 < n_qubits - 1 else 0
            alpha = np.random.uniform(0, 2*np.pi)
            qc.rxx(alpha, q1, q2)

    return qc


from scripts.agents.greedy_agent import GreedyMergeOptimizer
from tqdm import tqdm

MODE1 = MATCHGATE
MODE2 = MATCHGATE


n_qubitss = [5, 10, 15, 20, 25]
seeds = range(10)

for seed in tqdm(seeds):
    for n_qubits in tqdm(n_qubitss):
        n_gates = n_qubits * 5
        circuit = random_rz_rxx_circuit(n_qubits, n_gates, seed=seed)
        circuit_name = f"matchgateg2_{n_qubits}_{n_gates}_{seed}"

        #print(circuit)

        synthesis_config = {
            "MODE1": MODE1,  # or "solovay_kitaev"
            "MODE2": MODE2,  # or "solovay_kitaev"
            "EPS": 0.01,
        }

        model_config = {
            "ONLY_INTERACTIONS": True,
        }

        target_circuit_and_name = (circuit, circuit_name)

        #print number of qubits in circuit

        #print("Number of qubits in circuit:", circuit.num_qubits)

        optimizer = GreedyMergeOptimizer(
            synthesis_config=synthesis_config,
            target_circuit_and_name=target_circuit_and_name,
            model_config=model_config,
            KAK_SYNTHESIS_CACHE=None,
            U3_SYNTHESIS_CACHE=None
        )
        greedy_sequence, greedy_tcount = optimizer.optimize(verbose=False)
        optimizer.report()

