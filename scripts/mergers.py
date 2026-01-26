
import numpy as np
from typing import Tuple, List


from qiskit import QuantumCircuit

from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.circuit.library import UGate
from qiskit.circuit.library import UnitaryGate


def remove_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove all barriers from the circuit."""
    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for instr, qargs, cargs in circuit.data:
        if instr.name != "barrier":
            new_circuit.append(instr, qargs, cargs)
    return new_circuit



def remove_identities(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove all identity gates from the circuit."""
    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for instr, qargs, cargs in circuit.data:
        if instr.name != "id":
            new_circuit.append(instr, qargs, cargs)
    return new_circuit


#MERGERS
def merge_single_qubit_gates(input_circuit):
    """Merge logically adjacent single-qubit gates on the same qubit into a single U3 gate."""
    n_qubits = input_circuit.num_qubits
    merged_circuit = QuantumCircuit(n_qubits, input_circuit.num_clbits)

    # Temporary storage: accumulators per qubit
    accumulators = [QuantumCircuit(1) for _ in range(n_qubits)]
    qubit_dirty = [False] * n_qubits  # Whether accumulator is non-empty

    def flush(qubit_index):
        """Flush accumulator for one qubit into the merged circuit."""
        if not qubit_dirty[qubit_index]:
            return

        acc = accumulators[qubit_index]
        op = Operator(acc)

        if op.equiv(Operator(np.eye(2))):
            # Skip identity gates
            accumulators[qubit_index] = QuantumCircuit(1)
            qubit_dirty[qubit_index] = False
            return

        decomposer = OneQubitEulerDecomposer(basis='U3')
        u3_circuit = decomposer(op)
        instr, _, _ = u3_circuit.data[0]
        if instr.name != "u3":
            raise ValueError("Expected U3 gate after decomposition")
        theta, phi, lam = instr.params
        merged_circuit.u(theta, phi, lam, qubit_index)

        # Reset accumulator
        accumulators[qubit_index] = QuantumCircuit(1)
        qubit_dirty[qubit_index] = False

    # MAIN LOOP: process each instruction
    for instr, qargs, cargs in input_circuit.data:
        qubits_touched = [input_circuit.find_bit(q).index for q in qargs]

        if instr.num_qubits == 1:
            q_index = qubits_touched[0]
            accumulators[q_index].append(instr, [0])
            qubit_dirty[q_index] = True
        else:
            # Flush accumulators for the qubits involved *only*
            for q_index in qubits_touched:
                flush(q_index)
            merged_circuit.append(instr, qargs, cargs)

    # Final flush for all qubits
    for q in range(n_qubits):
        flush(q)

    return merged_circuit


def merge_contiguous_2q(
    circuit: QuantumCircuit,
    q0: int,
    q1: int,
    KAK_dec: bool = False,
) -> QuantumCircuit:
    """
    Merge only *contiguous* runs of gates acting solely on qubits (q0, q1).
    Flush only if a gate touches q0/q1 and another qubit outside this pair.
    """
    new_circ = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    buffer_2q: List[Tuple] = []

    def flush_buffer():
        nonlocal buffer_2q, new_circ
        if not buffer_2q:
            return

        contains_two_qubit = any(len(qargs) == 2 for _, qargs, _ in buffer_2q)

        if not contains_two_qubit:
            for instr, qargs, cargs in buffer_2q:
                new_circ.append(instr, qargs, cargs)
        elif len(buffer_2q) == 1:
            instr, qargs, cargs = buffer_2q[0]
            new_circ.append(instr, qargs, cargs)
        else:
            sub = QuantumCircuit(2)
            mapping = {q0: 0, q1: 1}
            for instr, qargs, cargs in buffer_2q:
                idxs = [mapping[circuit.find_bit(q).index] for q in qargs]
                sub.append(instr, idxs, [])
            U = Operator(sub)
            if not KAK_dec:
                merged_gate = UnitaryGate(U, label="merged₂q")
                new_circ.append(merged_gate, [q0, q1])
            else:
                KAK_circuit = QuantumCircuit(2)
                KAK_circuit.unitary(U, range(2))
                KAK_circuit = KAK_circuit.decompose()
                for inst, qargs, cargs in KAK_circuit.data:
                    local_idxs = [KAK_circuit.find_bit(q).index for q in qargs]
                    global_qubits = [q0 if i == 0 else q1 for i in local_idxs]
                    new_circ.append(inst, global_qubits, cargs)

        buffer_2q = []

    for instr, qargs, cargs in circuit.data:
        touched = {circuit.find_bit(q).index for q in qargs}

        if touched.issubset({q0, q1}):
            buffer_2q.append((instr, qargs, cargs))
        elif touched & {q0, q1} and not touched <= {q0, q1}:
            flush_buffer()
            new_circ.append(instr, qargs, cargs)
        else:
            new_circ.append(instr, qargs, cargs)

    flush_buffer()
    return new_circ


def merge_1q_into_2q(
    circuit: QuantumCircuit,
    q0: int,
    q1: int,
    reverse: bool = False,
) -> QuantumCircuit:
    """
    Merge 1q gates on q0 into the next (reverse=False) or previous (reverse=True)
    2q gate on (q0, q1), provided no interfering multi-qubit gates act on q0 in between.
    """
    n_qubits = circuit.num_qubits
    new_circ = QuantumCircuit(n_qubits, circuit.num_clbits)

    data = list(circuit.data)
    if reverse:
        data = list(reversed(data))

    buffer_1q: List[Tuple] = []

    def flush_buffer():
        nonlocal buffer_1q
        insert = reversed(buffer_1q) if reverse else buffer_1q
        for instr, qargs, cargs in insert:
            new_circ.append(instr, qargs, cargs)
        buffer_1q.clear()

    i = 0
    while i < len(data):
        instr, qargs, cargs = data[i]
        touched = {circuit.find_bit(q).index for q in qargs}

        if touched == {q0} and len(qargs) == 1:
            buffer_1q.append((instr, qargs, cargs))
            i += 1
            continue

        if touched == {q0, q1} and len(qargs) == 2:
            if buffer_1q:
                sub = QuantumCircuit(2)
                mapping = {q0: 0, q1: 1}

                if reverse:
                    # Append 2q gate first, then buffered 1q gates
                    idxs = [mapping[circuit.find_bit(q).index] for q in qargs]
                    sub.append(instr, idxs)
                    for instr_1q, qargs_1q, _ in buffer_1q:
                        idx = mapping[circuit.find_bit(qargs_1q[0]).index]
                        sub.append(instr_1q, [idx])
                else:
                    # Buffered 1q gates first, then 2q gate
                    for instr_1q, qargs_1q, _ in buffer_1q:
                        idx = mapping[circuit.find_bit(qargs_1q[0]).index]
                        sub.append(instr_1q, [idx])
                    idxs = [mapping[circuit.find_bit(q).index] for q in qargs]
                    sub.append(instr, idxs)

                U = Operator(sub)
                label = "merged_1q_2q" if not reverse else "merged_2q_1q"
                merged = UnitaryGate(U, label=label)
                new_circ.append(merged, [q0, q1])
                buffer_1q.clear()
            else:
                new_circ.append(instr, qargs, cargs)
            i += 1
            continue

        if touched & {q0} and not touched <= {q0, q1}:
            flush_buffer()
            new_circ.append(instr, qargs, cargs)
        else:
            new_circ.append(instr, qargs, cargs)

        i += 1

    flush_buffer()

    if reverse:
        new_circ.data = list(reversed(new_circ.data))

    return new_circ


def refine_merge(qc, q0, q1):
    qc = merge_single_qubit_gates(qc)
    qc = merge_1q_into_2q(qc, q0, q1)
    qc = merge_1q_into_2q(qc, q1, q0)
    qc = merge_1q_into_2q(qc, q1, q0, reverse=True)
    qc = merge_1q_into_2q(qc, q0, q1, reverse=True)
    return qc

def merge_contiguous_2q_and_refine(
    circuit: QuantumCircuit,
    q0: int,
    q1: int,
    KAK_dec: bool = False,
) -> QuantumCircuit:
    """
    Merge contiguous 2-qubit gates and refine the circuit by merging 1-qubit gates.
    """
    merged_circuit = merge_contiguous_2q(circuit, q0, q1, KAK_dec=KAK_dec)
    refined_circuit = refine_merge(merged_circuit, q0, q1)
    return refined_circuit


def apply_sequence_merge(qc, sequence, refine=True):
    """
    Apply a sequence of merge operations to the quantum circuit.
    The sequence is a list of tuples (q0, q1) indicating which qubits to merge.
    """
    if refine:
        for q0, q1 in sequence:
            #qc = merge_single_qubit_gates(qc)
            qc = merge_contiguous_2q_and_refine(qc, q0, q1)
    else:
        for q0, q1 in sequence:
            qc = merge_contiguous_2q(qc, q0, q1)
    return qc