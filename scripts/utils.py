import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

CLIFFORD_T_GATES = {
    'h', 's', 'sdg', 't', 'tdg', 'cx', 'x', 'y', 'z', 'cz', 'swap', 'id', 'barrier'
}

def compute_fidelity(circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> float:
    """Compute the fidelity between two quantum circuits."""
    U1 = Operator(circuit1).data
    U2 = Operator(circuit2).data
    return compute_fidelity_from_unitaries(U1, U2)

def compute_fidelity_from_unitaries(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute the fidelity between two unitaries.
    Here we compute the normalized process fidelity:
    F = |Tr(U1† U2)|^2 / d^2, where d is the dimension.
    """
    d = U1.shape[0]
    inner_product = np.trace(U1.conj().T @ U2)
    fidelity = np.abs(inner_product) ** 2 / (d ** 2)
    return fidelity

def fidelity_to_error(fidelity):
    if np.abs(1-fidelity)<1e-13: 
        return "-"
    return -np.log10(1 - fidelity)


def is_clifford_plus_t(circuit: QuantumCircuit) -> bool:
    """Return True if the circuit only contains Clifford+T gates."""
    return all(instr.name in CLIFFORD_T_GATES for instr, _, _ in circuit.data)