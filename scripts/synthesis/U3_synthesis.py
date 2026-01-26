from typing import Tuple, List, Optional


from qiskit import QuantumCircuit

from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer, generate_basic_approximations
from qiskit.circuit.library import UGate

from qiskit.transpiler.passes import SolovayKitaev


from pygridsynth.gridsynth import gridsynth_gates

# U3 stuff

from scripts.mergers import remove_barriers, remove_identities



from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.quantum_info import Operator


GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"


def extract_u3_params(gate):
    """Extract U3 parameters from any 1-qubit gate."""
    op = Operator(gate)
    decomposer = OneQubitEulerDecomposer(basis='U3')
    u3_circuit = decomposer(op)

    # Extract U3 parameters from the first instruction (should be a U3)
    instr, _, _ = u3_circuit.data[0]
    if not instr.name == "u3":
        raise ValueError("Decomposition did not return a U3 gate.")
    
    return instr.params  # [theta, phi, lambda]




def build_rz_from_gridsynth(gate_string):
    """
    Build a circuit from a grid synthesis string.
    Returns both the circuit and the matrix.
    """
    qc = QuantumCircuit(1)
    for g in gate_string:
        if g == "H":
            qc.h(0)
        elif g == "T":
            qc.t(0)
        elif g == "S":
            qc.s(0)
        elif g == "X":
            qc.x(0)
    matrix = Operator(qc).data
    return qc, matrix


def single_gridsynthetizer(angle, epsilon):
    """Generates a Rz gate using the Gridsynth method."""
    gates = gridsynth_gates(theta=angle, epsilon=epsilon)
    circuit, _ = build_rz_from_gridsynth(gates)
    return circuit, gates

def U3_gridsynthetizer(params, epsilon):
    """Starting from a the parameters corresponding to a U3 gate,
    decompose it into ZXZ gates and synthesize each layer using Gridsynth.
    Args:
        params (list): List of parameters for the U3 gate.
        epsilon (float): Precision parameter for Gridsynth.
    Returns:
        QuantumCircuit: The synthesized circuit."""
    final_circuit = QuantumCircuit(1)
    u_gate = UGate(*params)
    decomposer = OneQubitEulerDecomposer(basis='ZXZ')
    decomposed = decomposer(u_gate)

    for instr in decomposed.data:
        op, _, _ = instr
        angle = op.params[0]
        do_change_basis = (op.name == "rx")

        if do_change_basis:
            final_circuit.h(0)

        new_circuit, gates = single_gridsynthetizer(angle, epsilon)
        final_circuit = final_circuit.compose(new_circuit)

        if do_change_basis:
            final_circuit.h(0)

        final_circuit.barrier()

    return final_circuit, gates


def U3_SKdec(params, recursion_degree=2):
    initial_circuit = QuantumCircuit(1)
    u_gate = UGate(*params)
    initial_circuit.append(u_gate, [0])
    skd = SolovayKitaev(recursion_degree=recursion_degree)
    final_circuit = skd(initial_circuit)
    gates = final_circuit.count_ops()
    return final_circuit, gates

