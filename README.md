# Quantum Circuit Pre-Synthesis: Learning Local Edits to Reduce *T*-count



**Authors:**  
- Daniele Lizzio Bosco           1, 2, :envelope: 
- Lukasz Cincio                 3, 4
- Giuseppe Serra                1
- Marco Cerezo                  5, 4, :envelope:



1: Department of Mathematics, Computer Science and Physics, University of Udine, Udine, Italy

2: Department of Biology, University of Naples Federico II, Naples, Italy

3: Theoretical Division, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, USA

4: Quantum Science Center, Oak Ridge, TN 37931, USA

5: Information Sciences, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, USA

:envelope: Corresponding authors [lizziobosco.daniele@spes.uniud.it], [cerezo@lanl.gov]

**Read the paper here:**  
https://arxiv.org/abs/2601.19738

## Table of Contents
1. [Introduction](#Introduction)
2. [Reproducing the Experiments](#reproducing-the-experiments)
4. [Citation](#citation)


## Introduction

Compiling quantum circuits into Clifford+T gates is a central task for fault-tolerant quantum computing using stabilizer codes. In the near term, T gates will dominate the cost of fault tolerant implementations, and any reduction in the number of such expensive gates could mean the difference between being able to run a circuit or not. While exact synthesis is exponentially hard in the number of qubits, local synthesis approaches are commonly used to compile large circuits by decomposing them into substructures. However, composing local methods leads to suboptimal compilations in key metrics such as T-count or circuit depth, and their performance strongly depends on circuit representation. In this work, we address this challenge by proposing Q-PreSyn, a strategy that, given a set of local edits preserving circuit equivalence, uses a RL agent to identify effective sequences of such actions and thereby obtain circuit representations that yield a reduced T-count upon synthesis. Experimental results of our proposed strategy, applied on top of well-known synthesis algorithms, show up to a 20% reduction in T-count on circuits with up to 25 qubits, without introducing any additional approximation error prior to synthesis.

## Reproducing the Experiments
1. **Requirements:**  
    This project is based on Python ```3.11.5```.

2. **Setup Instructions:**  
    To set up the environment and install necessary dependencies (with venv):

   ```sh
   git clone https://github.com/Dan-LB/Quantum-Pre-Synthesis.git
   cd Quantum-Pre-Synthesis
   python3 -m venv qpresyn
   source qpresyn/bin/activate
   pip install -r requirements.txt
   ```

3. **Main results:**
    Each result in the Numerical Implementation section II.E can be reproduced by executing the corresponding file with
    ```sh
   python3 -m experiments.{exp_name}
   ```
   where ```exp_name``` is:
   * 1. General Random Circuit: ```general_random_circuit```
   * 1. (b) Scaling up to 25 qubits:  ```general_random_circuit_more_qubits```
   * 2. Linear Connectivity: ```linear_connectivity```
   * 3. Real-time Dynamics of the generalized Quantum Ising model: ```real_time_dynamics```
   * 4. Matchgate Circuits:  ```matchgate_synthesis```

In addition, the ```notebooks``` folder provides different example for greedy and RL agents, for the greedy refinement process, and for the matchgate synthesis.


## Citation
```bibtex
    @misc{lizziobosco2026quantumcircuitpresynthesislearning,
      title={Quantum Circuit Pre-Synthesis: Learning Local Edits to Reduce $T$-count}, 
      author={Daniele {Lizzio Bosco} and Lukasz Cincio and Giuseppe Serra and M. Cerezo},
      year={2026},
      eprint={2601.19738},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2601.19738}, 
}

```

