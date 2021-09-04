# Optimization of Quantum Clifford Circuit

## General Introduction / Overall Goal
In order to reduce hardware-level error rates, the modular architecture is proposed. The approach is to create a network of small independent quantum registers of few qubits, with connections capable of distributing entangled pairs between nodes. 

## State of the Art  & Key Problem
Experimentally, there have been significant advances in creating entanglement between modules. However, the infidelity of created Bell pairs is on the order of 10%, while noise due to local gates and measurements can be much lower than 1%.  Purification of the entanglement resource will thus be necessary before successfully employing it for fault-tolerant computation or communication. Numerous purification protocols have been proposed in literature, but there seems to lack a systematic comparison and optimization of purification circuits.

## Approach
Here, we develop a reinforcement learning algorithm in Julia to generate and find optimal purification circuits.
