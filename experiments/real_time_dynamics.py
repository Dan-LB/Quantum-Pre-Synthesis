import os
import random
import sys

from typing import Tuple, List, Optional
from qiskit import QuantumCircuit
from tqdm import tqdm

from scripts.agents.greedy_agent import GreedyMergeOptimizer
from scripts.agents.base_agent import CliffordTSynthesizer


from scripts.mergers import apply_sequence_merge

from copy import deepcopy
import itertools

from qiskit import qpy
from scripts.mergers import merge_single_qubit_gates


import csv

from qiskit.quantum_info import Operator

script_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(script_dir)

from qiskit import QuantumCircuit
import numpy as np

def analyze_custom_brickwork(qc):
    """
    Parse a 2k-qubit brickwork circuit with the pattern:
    
    Layer 0:        U, Even
    Middle layers:  Odd, U, Even
    Last layer:     Odd, U
    
    Also builds the equivalent QuantumCircuit for each layer.
    """
    num_qubits = qc.num_qubits
    k = num_qubits // 2

    all_instrs = list(qc.data)
    idx = 0

    structure = {
        "L": 0,
        "layers": []
    }

    layer_circuits = []

    def extract_u_layer(idx, layer_dict, layer_qc):
        for _ in range(num_qubits):
            instr = all_instrs[idx]

            qids = [qc.find_bit(q).index for q in instr.qubits]
            op = instr.operation

            # Store params only if single-qubit
            if op.num_qubits == 1:
                layer_dict["u_gates"][qids[0]] = op.params
            else:
                # Multi-qubit gates should not be assumed in U-layer
                layer_dict.setdefault("unexpected_multiq_in_u", []).append((op.name, qids))

            layer_qc.append(op, qids)
            idx += 1
        return idx


    def extract_even_layer(idx, layer_dict, layer_qc):
        for _ in range(k):
            instr = all_instrs[idx]
            qids = [qc.find_bit(q).index for q in instr.qubits]
            op = instr.operation

            layer_dict["even_pairs"][tuple(qids)] = op.name
            layer_qc.append(op, qids)
            idx += 1
        return idx


    def extract_odd_layer(idx, layer_dict, layer_qc):
        for _ in range(k):
            instr = all_instrs[idx]
            qids = [qc.find_bit(q).index for q in instr.qubits]
            op = instr.operation

            layer_dict["odd_pairs"][tuple(qids)] = op.name
            layer_qc.append(op, qids)
            idx += 1
        return idx


    # ---------------------------------------------------------
    # First layer: U, Even
    # ---------------------------------------------------------
    first_layer = {
        "u_gates": {},
        "even_pairs": {},
        "odd_pairs": {}
    }
    first_layer_qc = QuantumCircuit(num_qubits)

    idx = extract_u_layer(idx, first_layer, first_layer_qc)
    idx = extract_even_layer(idx, first_layer, first_layer_qc)

    structure["layers"].append(first_layer)
    layer_circuits.append(first_layer_qc)
    structure["L"] += 1

    # ---------------------------------------------------------
    # Middle layers: Odd, U, Even
    # Continue while enough instructions remain.
    # One middle-layer block consumes: k + num_qubits + k = num_qubits + 2k instructions
    # ---------------------------------------------------------
    remaining = len(all_instrs) - idx
    block_size = num_qubits + 2*k

    while remaining >= block_size:
        layer = {
            "u_gates": {},
            "even_pairs": {},
            "odd_pairs": {}
        }
        layer_qc = QuantumCircuit(num_qubits)

        idx = extract_odd_layer(idx, layer, layer_qc)
        idx = extract_u_layer(idx, layer, layer_qc)
        idx = extract_even_layer(idx, layer, layer_qc)

        structure["layers"].append(layer)
        layer_circuits.append(layer_qc)
        structure["L"] += 1

        remaining = len(all_instrs) - idx

    # ---------------------------------------------------------
    # Final layer: Odd, U
    # ---------------------------------------------------------
    final_layer = {
        "u_gates": {},
        "even_pairs": {},
        "odd_pairs": {}
    }
    final_layer_qc = QuantumCircuit(num_qubits)

    idx = extract_odd_layer(idx, final_layer, final_layer_qc)
    idx = extract_u_layer(idx, final_layer, final_layer_qc)

    structure["layers"].append(final_layer)
    layer_circuits.append(final_layer_qc)
    structure["L"] += 1

    return structure, layer_circuits

def combine_circuit_from_layers(layer_circuits):
    combined_qc = QuantumCircuit(layer_circuits[0].num_qubits)
    for layer_qc in layer_circuits:
        combined_qc = combined_qc.compose(layer_qc)
        print(layer_qc)
    return combined_qc


from scripts.agents.greedy_agent import GreedyMergeOptimizer

def optimize_layer_circuits(layer_circuits, synthesis_config, model_config,
                            KAK_SYNTHESIS_CACHE, U3_SYNTHESIS_CACHE):


    results = []
    merged_circuits = []

    for i, layer_qc in enumerate(layer_circuits):

        circuit_name = f"layer_{i}"
        target_circuit_and_name = (deepcopy(layer_qc), circuit_name)

        optimizer = GreedyMergeOptimizer(
            synthesis_config=synthesis_config,
            target_circuit_and_name=target_circuit_and_name,
            model_config=model_config,
        )

        synthesizer = CliffordTSynthesizer(mode1=synthesis_config["MODE1"],
                                   mode2=synthesis_config["MODE2"],
                                   epsilon=synthesis_config["EPS"],
                                   recursion_degree=synthesis_config["DEPTH"])

        # --- Compute initial T-count ---
        _, initial_tcounts = synthesizer(deepcopy(layer_qc))
        initial_tcount = sum(initial_tcounts)

        # --- Apply greedy optimization ---
        greedy_sequence, greedy_tcount = optimizer.optimize(verbose=False)

        # --- Final T-count ---
        merged_circuit = apply_sequence_merge(
            deepcopy(layer_qc),
            greedy_sequence,
            refine=True
        )
        merged_circuits.append(merged_circuit)



        # Store result
        results.append({
            "layer_index": i,
            "initial_tcount": initial_tcount,
            "final_tcount": greedy_tcount,
            "greedy_sequence": greedy_sequence,
            "optimized_circuit": merged_circuit
        })

    return results, merged_circuits

GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"

KAK = "kak"
BQSKit = "bqskit"


TYPES = ["T"]
SEEDS = range(28)
N_QUBITSS = [4, 6, 8]
Combinations = [
    (SOLOVAY_KITAEV, KAK),
    (GRIDSYNTH, BQSKit),
    (GRIDSYNTH, KAK),
]

def run_experiment(Q, TYPE, SEED, MODE1, MODE2, EPS=0.01):
    circuit_name = f"{TYPE}_Q_{Q}_num_10_id_{SEED}"
    qpy_path = f"experiments/real_dynamics_data/{circuit_name}.qpy"

    # ---------------------------
    # Load circuit
    # ---------------------------
    with open(qpy_path, "rb") as f:
        initial_circuit = qpy.load(f)[0]

    initial_circuit = merge_single_qubit_gates(initial_circuit)
    initial_circuit.data = [
        inst for inst in initial_circuit.data
        if inst[0].name != "set_statevector"
    ]

    synthesis_config = {
        "MODE1": MODE1,
        "MODE2": MODE2,
        "EPS": EPS,
        "DEPTH": 2,
    }

    model_config = {"ONLY_INTERACTIONS": True}

    # ---------------------------
    # Global greedy optimization
    # ---------------------------
    optimizer = GreedyMergeOptimizer(
        synthesis_config=synthesis_config,
        target_circuit_and_name=(deepcopy(initial_circuit), circuit_name),
        model_config=model_config,
    )

    synthesizer = CliffordTSynthesizer(mode1=synthesis_config["MODE1"],
                            mode2=synthesis_config["MODE2"],
                            epsilon=synthesis_config["EPS"],
                            recursion_degree=synthesis_config["DEPTH"])

    _, initial_tcounts = synthesizer(deepcopy(initial_circuit))
    initial_tcount = sum(initial_tcounts)

    _, global_greedy_tcount = optimizer.optimize(verbose=False)

    # ---------------------------
    # Layer-wise optimization
    # ---------------------------
    structure, layer_circuits = analyze_custom_brickwork(initial_circuit)

    results, merged_circuits = optimize_layer_circuits(
        layer_circuits,
        synthesis_config=synthesis_config,
        model_config=model_config,
        KAK_SYNTHESIS_CACHE=None,
        U3_SYNTHESIS_CACHE=None,
    )

    final_layerwise_tcount = sum(r["final_tcount"] for r in results)

    final_merged_circuit = combine_circuit_from_layers(
        [r["optimized_circuit"] for r in results]
    )
    final_merged_circuit = merge_single_qubit_gates(final_merged_circuit)

    _, final_tcounts_checked = synthesizer(deepcopy(final_merged_circuit))
    final_tcount_check = sum(final_tcounts_checked)

    # Re-check via GreedyMergeOptimizer
    opt2 = GreedyMergeOptimizer(
        synthesis_config=synthesis_config,
        target_circuit_and_name=(deepcopy(initial_circuit), circuit_name),
        model_config=model_config,
    )
    _, reoptimized_tcount = opt2.optimize(verbose=False)

    if final_tcount_check != reoptimized_tcount:
        print(
            f"Warning: layerwise T-count {final_tcount_check} does not match "
            f"reoptimized value {reoptimized_tcount}."
        )

    # ---------------------------
    # Return results
    # ---------------------------
    return {
        "Q": Q,
        "TYPE": TYPE,
        "SEED": SEED,
        "MODE1": MODE1,
        "MODE2": MODE2,
        "EPS": EPS,
        "initial_tcount": initial_tcount,
        "global_greedy_tcount": global_greedy_tcount,
        "final_layerwise_tcount": final_layerwise_tcount,
        "final_final_tcount": reoptimized_tcount,
        "num_layers": structure["L"],
    }, merged_circuits


# ------------------------------------------------------------
# MAIN EXPERIMENT LOOP (NOW MATCHING THE FIRST SCRIPT)
# ------------------------------------------------------------
csv_path = "real_time_dynamics.csv"
file_exists = os.path.isfile(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "Q",
            "TYPE",
            "SEED",
            "MODE1",
            "MODE2",
            "EPS",
            "initial_tcount",
            "global_greedy_tcount",
            "final_layerwise_tcount",
            "final_final_tcount",
            "num_layers",
        ])

    experiment_list = [
        (Q, TYPE, SEED, MODE1, MODE2)
        for Q in N_QUBITSS
        for TYPE in TYPES
        for SEED in SEEDS
        for (MODE1, MODE2) in Combinations
    ]

    print(f"Total experiments: {len(experiment_list)}")

    for Q, TYPE, SEED, MODE1, MODE2 in tqdm(experiment_list, desc="Running iMPS experiments"):
        print(f"Running Q={Q}, TYPE={TYPE}, SEED={SEED}, MODE1={MODE1}, MODE2={MODE2}")

        result, merged_circuits = run_experiment(
            Q=Q,
            TYPE=TYPE,
            SEED=SEED,
            MODE1=MODE1,
            MODE2=MODE2,
            EPS=0.01,
        )

        writer.writerow(result.values())
        f.flush()
