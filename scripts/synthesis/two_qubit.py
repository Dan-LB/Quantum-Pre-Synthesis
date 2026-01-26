from typing import Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from scripts.synthesis.synthesis_util import unitary_to_cache_key

from bqskit import compile
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit

from scripts.utils import compute_fidelity_from_unitaries

KAK = "kak"
BQSKit = "bqskit"


def KAK_2qubits(input_circuit: QuantumCircuit,
                  KAK_SYNTHESIS_CACHE: dict[Tuple[Tuple[float, float], ...], QuantumCircuit] = None) -> QuantumCircuit:
    """Given a 2-qubit circuit, decompose it into KAK form using memoization."""
    op = Operator(input_circuit)
    unitary = op.data

    if KAK_SYNTHESIS_CACHE is not None:
        key = unitary_to_cache_key(unitary)
        if key in KAK_SYNTHESIS_CACHE:
            return KAK_SYNTHESIS_CACHE[key]

    # Perform decomposition if not in cache
    temp_circ = QuantumCircuit(2)
    temp_circ.unitary(op, [0, 1])
    decomp = temp_circ.decompose()

    # Store in cache
    if KAK_SYNTHESIS_CACHE is not None:
        KAK_SYNTHESIS_CACHE[key] = decomp
    return decomp


def BQSKit_2qubits(input_circuit: QuantumCircuit,
                   KAK_SYNTHESIS_CACHE: dict[Tuple[Tuple[float, float], ...], QuantumCircuit] = None) -> QuantumCircuit:
    """Given a 2-qubit circuit, optimize it using BQSKit."""
    op = Operator(input_circuit)
    unitary = op.data
    if KAK_SYNTHESIS_CACHE is not None:
        key = unitary_to_cache_key(unitary)
        if key in KAK_SYNTHESIS_CACHE:
            return KAK_SYNTHESIS_CACHE[key]

    # Perform optimization using BQSKit
    bqskit_circuit = qiskit_to_bqskit(input_circuit)
    out_circuit = compile(bqskit_circuit, optimization_level=1)
    qiskit_circuit = bqskit_to_qiskit(out_circuit)
    decomp = qiskit_circuit.decompose()

    if KAK_SYNTHESIS_CACHE is not None:
        KAK_SYNTHESIS_CACHE[key] = decomp

    return decomp



from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

# -------------------------------------------------
# Pauli definitions
# -------------------------------------------------
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Kronecker products
def kron(*mats):
    out = np.array([1.0], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out

# Two-qubit operators
XX = kron(X, X)
Z1 = kron(I, Z)
Z2 = kron(Z, I)

# -------------------------------------------------
# Gate constructors
# -------------------------------------------------
def Rxx(theta: float) -> np.ndarray:
    """Two-qubit XX rotation."""
    return expm(1j * theta * XX)

def Rz_pair(phi1: float, phi2: float) -> np.ndarray:
    """Product of single-qubit Z rotations."""
    return expm(1j * phi1 * Z1) @ expm(1j * phi2 * Z2)

def matchgate_unitary(params: np.ndarray) -> np.ndarray:
    """
    Build the 2-qubit matchgate unitary from 6 parameters:
      params = [alpha1, alpha2, beta11, beta21, beta12, beta22]
    """
    alpha1, alpha2, beta11, beta21, beta12, beta22 = params
    L1 = Rxx(alpha1)
    L2 = Rz_pair(beta11, beta21)
    L3 = Rxx(alpha2)
    L4 = Rz_pair(beta12, beta22)
    return L4 @ L3 @ L2 @ L1

# -------------------------------------------------
# Fitting routine
# -------------------------------------------------
def residual_vec(params: np.ndarray, U_target: np.ndarray) -> np.ndarray:
    """Residual between predicted and target unitaries (real+imag flattened)."""
    U_pred = matchgate_unitary(params)
    res = (U_pred - U_target).ravel()
    return np.concatenate([res.real, res.imag])

def fit_matchgate_parameters(U_target: np.ndarray,
                             initial_guess: np.ndarray | None = None) -> np.ndarray:
    """Fit the six parameters of a 2-qubit matchgate to a target unitary."""
    if initial_guess is None:
        initial_guess = np.zeros(6, dtype=float)
    res = least_squares(residual_vec, initial_guess, args=(U_target,),
                        method='lm', max_nfev=10000)
    params_fit = np.mod(res.x + np.pi, 2 * np.pi) - np.pi  # wrap to (-pi, pi]
    return params_fit

# -------------------------------------------------
# Qiskit helper
# -------------------------------------------------
def matchgate_2qubits(input_circuit: QuantumCircuit,
                                  verbose: bool = False) -> QuantumCircuit:
    """
    Fit a 2-qubit circuit U_target to the canonical 6-parameter matchgate form.
    Returns a new QuantumCircuit(2) implementing that decomposition.
    """
    U_target = Operator(input_circuit).data
    params = fit_matchgate_parameters(U_target)

    if verbose:
        print("Recovered parameters (α₁, α₂, β₁₁, β₂₁, β₁₂, β₂₂):")
        print(np.round(params, 6))

    circ = QuantumCircuit(2)
    # params = [alpha1, alpha2, beta11, beta21, beta12, beta22]
    circ.rxx(-2.0 * params[0], 0, 1)
    circ.rz(-2.0 * params[2], 0)
    circ.rz(-2.0 * params[3], 1)
    circ.rxx(-2.0 * params[1], 0, 1)
    circ.rz(-2.0 * params[4], 0)
    circ.rz(-2.0 * params[5], 1)


    # Verify fidelity
    U_fit = Operator(circ).data
    fid = np.abs(np.trace(np.conjugate(U_fit.T) @ U_target)) / 4
    if verbose:
        print(f"Fidelity between target and fitted circuit: {fid:.12f}")

    return circ



