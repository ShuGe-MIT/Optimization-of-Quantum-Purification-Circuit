# QuantumClifford.jl

```@meta
DocTestSetup = quote
    using QuantumClifford
end
```

A Julia package for working with quantum stabilizer states and Clifford circuits
that act on them. It uses the tableaux formalism[^1] with the destabilizer improvments[^2].

[^1]: [gottesman1998heisenberg](@cite)

[^2]: [aaronson2004improved](@cite)

Works efficiently with
[pure](@ref Stabilizers) and
[mixed stabilizer](@ref Mixed-Stabilizer-States)
states of thousands of qubits
as well as
[sparse or dense Clifford operations](@ref Clifford-Operators)
acting upon them.

Provides
[canonicalization](@ref Canonicalization-of-Stabilizers),
[projection](@ref Projective-Measurements), and
[generation](@ref Generating-a-Pauli-Operator-with-Stabilizer-Generators) operations,
as well as
[partial traces](@ref Partial-Traces).

```jldoctest
julia> P"X" * P"Z"
-iY

julia> P"X" ⊗ P"Z"
+ XZ

julia> S"-XX
         +ZZ"
- XX
+ ZZ

julia> CNOT * S"-XX
                +ZZ"
- X_
+ _Z
```
