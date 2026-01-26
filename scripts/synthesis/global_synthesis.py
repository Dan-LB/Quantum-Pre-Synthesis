
from typing import Tuple, List, Optional

#KAK_SYNTHESIS_CACHE: dict[Tuple[Tuple[float, float], ...], QuantumCircuit] = {}

from qiskit import QuantumCircuit


GRIDSYNTH = "gridsynth"
SOLOVAY_KITAEV = "solovay_kitaev"

KAK = "kak"
BQSKit = "bqskit"


from scripts.synthesis.U3_synthesis import U3_SKdec, U3_gridsynthetizer, extract_u3_params
from scripts.synthesis.two_qubit import KAK_2qubits, BQSKit_2qubits
from scripts.utils import  is_clifford_plus_t
from scripts.mergers import merge_single_qubit_gates, remove_barriers, remove_identities

#U3_SYNTHESIS_CACHE: dict[Tuple[float, float, float], Tuple[QuantumCircuit, List[str]]] = {}

from typing import Optional, List, Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit



def decompose_2qubits(circuit: QuantumCircuit,
                      MODE = KAK,
                      KAK_SYNTHESIS_CACHE: dict[Tuple[Tuple[float, float], ...], QuantumCircuit] = None) -> QuantumCircuit:

    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for instr, qargs, cargs in circuit.data:
        if instr.num_qubits == 2:
            # Get qubit indices
            q0 = circuit.find_bit(qargs[0]).index
            q1 = circuit.find_bit(qargs[1]).index

            # Build a 2-qubit circuit for this gate
            subcircuit = QuantumCircuit(2)
            # Insert the gate with correct indices in the 2q subcircuit
            subcircuit.append(instr, [0, 1])
            if MODE == KAK:
                dec = KAK_2qubits(subcircuit, KAK_SYNTHESIS_CACHE)
            elif MODE == BQSKit:
                dec = BQSKit_2qubits(subcircuit, KAK_SYNTHESIS_CACHE)
            else:
                raise ValueError(f"Unknown decomposition mode: {MODE}")

            # Append synthesized gate back to new_circuit on correct qubits
            for sub_instr, sub_qargs, sub_cargs in dec.data:
                if sub_instr.num_qubits == 1:
                    q_sub = dec.find_bit(sub_qargs[0]).index
                    q_global = q0 if q_sub == 0 else q1
                    new_circuit.append(sub_instr, [q_global], sub_cargs)
                elif sub_instr.num_qubits == 2:
                    # Only CNOTs should appear
                    ctrl = dec.find_bit(sub_qargs[0]).index
                    tgt = dec.find_bit(sub_qargs[1]).index
                    q_ctrl = q0 if ctrl == 0 else q1
                    q_tgt = q0 if tgt == 0 else q1
                    new_circuit.cx(q_ctrl, q_tgt)
                else:
                    raise ValueError(f"Unexpected instruction with {sub_instr.num_qubits} qubits in synthesized circuit")
        else:
            # Keep single-qubit (or other) gates untouched
            new_circuit.append(instr, qargs, cargs)

    return new_circuit

def synthesize_1qubit_gates(input_circuit: QuantumCircuit,
                             MODE = GRIDSYNTH,
                             to_remove_barriers: bool = True,
                             to_remove_identities: bool = True,
                             qubits_list: Optional[List[int]] = None,
                             U3_SYNTHESIS_CACHE: Optional[dict[Tuple[float, float, float], Tuple[QuantumCircuit, List[str]]]] = None,
                             epsilon: float = 1e-3,
                             recursion_degree: int = 2) -> QuantumCircuit:
    """
    Synthesize a circuit by replacing each 1-qubit gate with its Clifford+T approximation.
    If `qubits_list` is specified, synthesis is applied only to those qubits.
    """
    n_qubits = input_circuit.num_qubits
    synthesized_circuit = QuantumCircuit(n_qubits)

    t_count_counters = [0] * n_qubits

    if to_remove_barriers:
        input_circuit = remove_barriers(input_circuit)
    if to_remove_identities:
        input_circuit = remove_identities(input_circuit)

    for instr, qargs, cargs in input_circuit.data:
        if instr.num_qubits == 1 and instr.name != "barrier":
            qubit_index = input_circuit.find_bit(qargs[0]).index

            if qubits_list is None or qubit_index in qubits_list:
                # Convert instruction to a single-qubit QuantumCircuit
                temp_qc = QuantumCircuit(1)
                temp_qc.append(instr, [0])

                theta, phi, lam = extract_u3_params(temp_qc)
                rounded_params = tuple(round(x, 10) for x in (theta, phi, lam))

                if U3_SYNTHESIS_CACHE is None:
                    U3_SYNTHESIS_CACHE = {}
                if rounded_params in U3_SYNTHESIS_CACHE:
                    approx, gates = U3_SYNTHESIS_CACHE[rounded_params]

                else:
                    if MODE == GRIDSYNTH:
                        approx, gates = U3_gridsynthetizer([theta, phi, lam], epsilon)
                    elif MODE == SOLOVAY_KITAEV:
                        approx, gates = U3_SKdec([theta, phi, lam], recursion_degree=recursion_degree)

                    U3_SYNTHESIS_CACHE[rounded_params] = (approx, gates)

                if MODE == GRIDSYNTH:
                    t_count = gates.count('T')
                elif MODE == SOLOVAY_KITAEV:
                    t_count = gates.get('t', 0) + gates.get('tdg', 0)
                t_count_counters[qubit_index] += t_count

                # Compose the synthesized circuit on the correct qubit
                synthesized_circuit.compose(approx, qubits=[qubit_index], inplace=True)
                #print(f"Time composed synthesized circuit: {time.time() - current_time:.6f} seconds   ")    
            else:
                synthesized_circuit.append(instr, qargs, cargs)
        else:
            synthesized_circuit.append(instr, qargs, cargs)

    #print(t_count_counters)
    #print(f"The cache contains {len(U3_SYNTHESIS_CACHE)} entries.")
    return synthesized_circuit, t_count_counters



def clifford_t_synthesis(input_circuit,
                         MODE1=GRIDSYNTH,
                         MODE2=KAK,
                         additional_1q_merge=True, to_remove_barriers=True,
                         to_remove_identities=True, verbose=False,
                         KAK_SYNTHESIS_CACHE: Optional[dict[Tuple[Tuple[float, float], ...], QuantumCircuit]] = None,
                         U3_SYNTHESIS_CACHE: Optional[dict[Tuple[float, float, float], Tuple[QuantumCircuit, List[str]]]] = None,
                         epsilon=1e-3,
                         recursion_depth=2) -> Tuple[QuantumCircuit, List[int]]:
    """
    Synthesize a circuit by replacing each 1-qubit gate with its Clifford+T approximation.
    """

    if to_remove_barriers:
        input_circuit = remove_barriers(input_circuit)
    if to_remove_identities:
        input_circuit = remove_identities(input_circuit)


    #print(f"I am using MODE2 as {MODE2}")
    synthesized_circuit = decompose_2qubits(input_circuit,
                                            MODE=MODE2,
                                            KAK_SYNTHESIS_CACHE=KAK_SYNTHESIS_CACHE)

    if additional_1q_merge:
        synthesized_circuit = merge_single_qubit_gates(synthesized_circuit)

    if MODE1 == GRIDSYNTH:
        synthesized_circuit, t_list = synthesize_1qubit_gates(synthesized_circuit, 
                                                              MODE=GRIDSYNTH,
                                                              to_remove_barriers=to_remove_barriers,
                                                              to_remove_identities=to_remove_identities,
                                                              U3_SYNTHESIS_CACHE=U3_SYNTHESIS_CACHE,
                                                              epsilon=epsilon,
                                                              )

    if MODE1 == SOLOVAY_KITAEV:
        synthesized_circuit, t_list = synthesize_1qubit_gates(synthesized_circuit,
                                                              MODE=SOLOVAY_KITAEV,
                                                              to_remove_barriers=to_remove_barriers,
                                                              to_remove_identities=to_remove_identities,
                                                              U3_SYNTHESIS_CACHE=U3_SYNTHESIS_CACHE,
                                                              recursion_degree=recursion_depth,
                                                              )

    if verbose:
        assert False, "Verbose mode is not implemented."
    
    if remove_barriers:
        synthesized_circuit = remove_barriers(synthesized_circuit)
    return synthesized_circuit, t_list


# MATCHGATE-BASED SYNTHESIS FOR GENERAL CIRCUITS

from pygridsynth.gridsynth import gridsynth_gates
from qiskit.circuit.library import RXXGate


def compile_angle_to_string(angle, precision):
    gatelist = gridsynth_gates(angle, precision)
    #print(f"Gates: {gatelist}")
    return gatelist


def ryx_via_rxx_qiskit(qc, alpha):
    # apply U† first (U = Rz(pi/2) on qubit 0)
    qc.rz(-np.pi/2, 0)
    qc.append(RXXGate(alpha), [0, 1])
    # then apply U
    qc.rz(np.pi/2, 0)
    return qc

def build_twoqubit_from_Rz_string(gate_string):
    """
    Build a 2-qubit circuit corresponding to your matchgate mapping.
    gate_string: str (e.g. "HSTX...")
    """
    qc = QuantumCircuit(2)
    
    for g in gate_string:
        if g == "H":
            # (S^2 ⊗ I) @ Ryx
            qc.z(0)   # since S^2 = Z
            qc = ryx_via_rxx_qiskit(qc, np.pi/2)  # Ryx² equivalent
        elif g == "S":
            qc.s(0)  # S ⊗ I
        elif g == "T":
            qc.t(0)  # T ⊗ I
        elif g == "W":
            continue  # No operation needed in circuit
        elif g == "X":
            qc.z(0)   # Z⊗I
            qc = ryx_via_rxx_qiskit(qc, np.pi)  # Ryx² equivalent
        elif g == "I":
            continue  # Identity, no operation needed
        else:
            raise ValueError(f"Unknown gate symbol: {g}")
    
    return qc

def build_twoqubit_from_Rxx_string(gate_string):
    """
    Build a 2-qubit circuit corresponding to your matchgate mapping.
    gate_string: str (e.g. "HSTX...")
    """
    qc = QuantumCircuit(2)
    qc.s(0)  # Initial S on qubit 0 to match the mapping
    #Rxx · (Z⊗I)
    qc.rxx(np.pi/2, 0, 1)   # Rxx
    qc.z(0)   # since S^2 = Z

    for g in gate_string:
        if g == "H":
            qc.rxx(np.pi/2, 0, 1)   # Rxx
            qc.z(0)   # since S^2 = Z
        elif g == "S":
            qc.s(0)  # S ⊗ I
        elif g == "T":
            qc.t(0)  # T ⊗ I
        elif g == "W":
            continue  # No operation needed in circuit
        elif g == "X":
            qc.sdg(0)  # S†⊗I
            qc.rxx(np.pi, 0, 1)   # Rxx²
            qc.s(0)  # S⊗I
        elif g == "I":
            continue  # Identity, no operation needed
        else:
            raise ValueError(f"Unknown gate symbol: {g}")

    qc.rxx(np.pi/2, 0, 1)   # Rxx
    qc.z(0)   # since S^2 = Z
    qc.sdg(0)  # Final S on qubit 0 to match the mapping
    return qc



def decompose_general_rz_rxx_circuit(qc, precision=0.01):
    """
    Replace all Rz and Rxx gates in a general circuit with matchgate sequences.
    """

    # OTHERWISE, TO DO: DECOMPOSE U2 GATES WITH ... AND EVALUATE IF IT IS OK

    n_qubits = qc.num_qubits
    new_qc = QuantumCircuit(n_qubits)

    for inst, qargs, cargs in qc.data:
        # --- Handle RZ gates ---
        if inst.name == "rz":
            alpha = inst.params[0]
            q = qc.find_bit(qargs[0]).index   # global qubit index

            gridsynth_string = compile_angle_to_string(alpha, precision)
            # Suppose merge is happening on pair (q0, q1)
            q_main, q_aux = q0, q1
            local_qc = build_twoqubit_from_Rz_string(gridsynth_string)

            for sub_inst, sub_qargs, sub_cargs in local_qc.data:
                if sub_inst.num_qubits == 1:
                    q_sub = local_qc.find_bit(sub_qargs[0]).index
                    q_global = q_main if q_sub == 0 else q_aux
                    new_qc.append(sub_inst, [q_global], sub_cargs)
                elif sub_inst.num_qubits == 2:
                    q0_sub = local_qc.find_bit(sub_qargs[0]).index
                    q1_sub = local_qc.find_bit(sub_qargs[1]).index
                    q0_global = q_main if q0_sub == 0 else q_aux
                    q1_global = q_main if q1_sub == 0 else q_aux
                    new_qc.append(sub_inst, [q0_global, q1_global], sub_cargs)


        # --- Handle RXX gates ---
        elif inst.name == "rxx":
            alpha = inst.params[0]
            q0 = qc.find_bit(qargs[0]).index
            q1 = qc.find_bit(qargs[1]).index
            if q1 < q0:
                q0, q1 = q1, q0

            gridsynth_string = compile_angle_to_string(alpha, precision)
            local_qc = build_twoqubit_from_Rxx_string(gridsynth_string)

            for sub_inst, sub_qargs, sub_cargs in local_qc.data:
                if sub_inst.num_qubits == 1:
                    q_sub = local_qc.find_bit(sub_qargs[0]).index
                    q_global = q0 if q_sub == 0 else q1
                    new_qc.append(sub_inst, [q_global], sub_cargs)
                elif sub_inst.num_qubits == 2:
                    new_qc.append(sub_inst, [q0, q1], sub_cargs)
                else:
                    raise ValueError(f"Unsupported gate {sub_inst.name} in subcircuit")

        # --- Handle U3 gates ---
        elif inst.name == "u":
            theta, phi, lam = inst.params
            q = qc.find_bit(qargs[0]).index

            # If U3(0, a, a), replace with Rz(2a)
            if np.isclose(theta, 0):
                new_qc.rz(phi + lam, q)    # general θ=0
            else:
                raise ValueError(f"I tried to decompose a U3 as Rz, but it was not of the form U3(0,a,b). Params: theta={theta}, phi={phi}, lam={lam}")

        else:
            assert False, f"Unsupported gate {inst.name} in original circuit.\nGate info: {inst}, qargs: {qargs}, cargs: {cargs}"

    return new_qc



from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def decompose_2qubits_into_matchgate(
    circuit: QuantumCircuit,
    MATCHGATE_CACHE: dict[str, QuantumCircuit] = None,
    verbose: bool = False
) -> QuantumCircuit:
    """
    Replace every 2-qubit gate in a circuit by its minimal matchgate decomposition.

    Parameters
    ----------
    circuit : QuantumCircuit
        Input Qiskit circuit (can contain arbitrary gates).
    MATCHGATE_CACHE : dict[str, QuantumCircuit], optional
        Cache for already-synthesized 2-qubit unitaries, keyed by a hash of the unitary matrix.
    verbose : bool
        If True, prints diagnostic messages.

    Returns
    -------
    QuantumCircuit
        A new circuit where all 2-qubit gates have been replaced by their 4-layer matchgate decompositions.
    """

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    import numpy as np
    import hashlib
    from scripts.synthesis.two_qubit import matchgate_2qubits

    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    if MATCHGATE_CACHE is None:
        MATCHGATE_CACHE = {}

    for instr, qargs, cargs in circuit.data:
        n_q = instr.num_qubits

        # Handle 2-qubit gates
        if n_q == 2 and instr.name != "rz" and instr.name != "rxx":
            q0 = circuit.find_bit(qargs[0]).index
            q1 = circuit.find_bit(qargs[1]).index

            # Build a subcircuit containing only this instruction
            subcircuit = QuantumCircuit(2)
            subcircuit.append(instr, [0, 1])

            # Compute a compact hash of the unitary to use for caching
            try:
                U = Operator(subcircuit).data
                key = hashlib.sha1(np.round(U, 8).tobytes()).hexdigest()
            except Exception as e:
                key = f"{instr.name}_{instr.params}"

            # Check cache
            if key in MATCHGATE_CACHE:
                dec = MATCHGATE_CACHE[key]
                if verbose:
                    print(f"[Cache hit] Using stored matchgate decomposition for {instr.name}.")
            else:
                # Perform matchgate synthesis
                dec = matchgate_2qubits(subcircuit)
                MATCHGATE_CACHE[key] = dec
                if verbose:
                    print(f"[Decompose] {instr.name} on qubits ({q0}, {q1}) synthesized to matchgate form.")

            # Append decomposition back to the correct qubits
            for sub_instr, sub_qargs, sub_cargs in dec.data:
                if sub_instr.num_qubits == 1:
                    q_sub = dec.find_bit(sub_qargs[0]).index
                    q_global = q0 if q_sub == 0 else q1
                    new_circuit.append(sub_instr, [q_global], sub_cargs)
                elif sub_instr.num_qubits == 2:
                    q0_sub = dec.find_bit(sub_qargs[0]).index
                    q1_sub = dec.find_bit(sub_qargs[1]).index
                    q0_global = q0 if q0_sub == 0 else q1
                    q1_global = q0 if q1_sub == 0 else q1
                    new_circuit.append(sub_instr, [q0_global, q1_global], sub_cargs)
                else:
                    raise ValueError(f"Unexpected gate with {sub_instr.num_qubits} qubits in matchgate decomposition.")

        else:
            # Keep 1-qubit or measurement operations untouched
            new_circuit.append(instr, qargs, cargs)

    return new_circuit




def clifford_t_synthesis_matchgate(input_circuit: QuantumCircuit,
                                   additional_1q_merge: bool = True,
                                   to_remove_barriers: bool = True,
                                   to_remove_identities: bool = True,
                                   verbose: bool = False,
                                   U3_SYNTHESIS_CACHE: Optional[dict[Tuple[float, float, float], Tuple[QuantumCircuit, List[str]]]] = None,
                                   epsilon: float = 1e-3,
                                   recursion_depth: int = 2
                                   ) -> Tuple[QuantumCircuit, List[int]]:
    """
    Synthesize a circuit by replacing each 1-qubit gate with its Clifford+T approximation,
    """

    # --- Step 0: Preprocess circuit ---
    if to_remove_barriers:
        input_circuit = remove_barriers(input_circuit)
    if to_remove_identities:
        input_circuit = remove_identities(input_circuit)

    # --- Step 1: Decompose all 2-qubit gates into minimal matchgate form ---
    decomposed_circuit = decompose_2qubits_into_matchgate(input_circuit)
    decomposed_circuit = merge_single_qubit_gates(decomposed_circuit)
    #print(decomposed_circuit)
    synthesized_circuit = decompose_general_rz_rxx_circuit(decomposed_circuit, precision=epsilon)
    t_count = sum(1 for instr, _, _ in synthesized_circuit.data if instr.name in ("t", "tdg"))
    t_list = [t_count] * synthesized_circuit.num_qubits


    return synthesized_circuit, t_list
