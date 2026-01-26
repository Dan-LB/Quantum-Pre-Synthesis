from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit

def get_possible_merges(qc: QuantumCircuit, only_interactions: bool = False):
    """
    Return possible 2-qubit merges in a circuit.

    Parameters:
        qc (QuantumCircuit): A Qiskit quantum circuit.
        only_interactions (bool): If True, return only pairs of qubits that
                                  interact in the circuit (via multi-qubit gates).
                                  If False, return all possible pairs.

    Returns:
        List of qubit pairs (tuples) that can be merged.
    """
    n_qubits = qc.num_qubits

    if not only_interactions:
        # All possible 2-qubit combinations
        merg_set = set(combinations(range(n_qubits), 2))
        list_merg = list(merg_set)
        return list_merg
    
    # Only pairs of qubits that interact through multi-qubit gates
    interacting_pairs = set()

    for instr, qargs, _ in qc.data:
        qubit_indices = [qc.qubits.index(q) for q in qargs]
        if len(qubit_indices) >= 2:
            for pair in combinations(qubit_indices, 2):
                interacting_pairs.add(tuple(sorted(pair)))  # Ensure consistent ordering

    interacting_pairs = list(interacting_pairs)
    return interacting_pairs

def refine_from_coupling_map(qc: QuantumCircuit, coupling_map: list[tuple[int, int]]) -> QuantumCircuit:
    """
    Return a new QuantumCircuit with 2-qubit gates removed if their qubit pair is not in the coupling map.

    Args:
        qc (QuantumCircuit): Original quantum circuit.
        coupling_map (list of tuple): Allowed qubit pairs (undirected).

    Returns:
        QuantumCircuit: Refined circuit with only allowed 2-qubit gates.
    """
    allowed_pairs = set(tuple(sorted(pair)) for pair in coupling_map)
    refined_qc = QuantumCircuit(qc.num_qubits)

    for instr, qargs, cargs in qc.data:
        qubit_indices = [qc.qubits.index(q) for q in qargs]

        if len(qubit_indices) == 2:
            pair = tuple(sorted(qubit_indices))
            if pair in allowed_pairs:
                refined_qc.append(instr, qargs, cargs)
        else:
            # Keep single-qubit or higher-order gates
            refined_qc.append(instr, qargs, cargs)

    return refined_qc
