Optimized Quantum Entanglement

`qevo.py` provides a small python library for the creation and optimization of
quantum entanglement purification circuits.

`Clifford.jl` is a higly-optimized julia library for enumerating and studying
the Clifford group of multiple qubits.

`examples` contains multiple notebooks showcasing the use of these libraries.
Some of these files were used in the writing of the related paper.

- `Example` shows how to run a simple optimization for the creation of
  purification circuits.

- `Compare_Regimes` shows that for different parameter regimes (i.e. error
  models) different circuits are better.

- `HotCold` shows how to augment the library to work with custom error models
  and hardware architectures. In this case we optimize for a register that has
  only one "hot" qubit (a communication qubit capable of establishing initial remote
  entanglement).

- `OptimizeHashingYield` shows how to optimize for the hashing yield (defined
  only for perfect local operations). It is of great theoretical interest in
  the study of assymptotic circuits, but it is less useful in our case of small
  circuits optimized for operational errors.

- `Structure_ParallelNaiveCoarseDividing` and `julia-subgroup` are used to
  enumerate and study the group structure of the Clifford/Permutation
  operations used in the purification circuits.

See [qevo.krastanov.org](https://qevo.krastanov.org) for visualizations and
comparisons of circuits generated by this software.