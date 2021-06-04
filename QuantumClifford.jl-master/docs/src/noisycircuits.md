# Simulation of Noisy Clifford Circuits

```@meta
DocTestSetup = quote
    using QuantumClifford
    using QuantumClifford.Experimental.NoisyCircuits
end
CurrentModule = QuantumClifford.Experimental.NoisyCircuits
```

We have experimental support for simulation of noisy Clifford circuirts which can be imported with `using QuantumClifford.Experimental.NoisyCircuits`.

Both [Monte Carlo](@ref noisycircuits_mc) and [Perturbative Expansion](@ref noisycircuits_perturb) approaches are supported. When performing a perturbative expansion in the noise parameter, the expansion can optionally be performed symbolically, to arbitrary high orders.

Multiple [notebooks with examples](@ref tutandpub) are also available.
For instance, see this tutorial on [entanglement purification for many examples](https://github.com/Krastanov/QuantumClifford.jl/blob/master/docs/src/notebooks/Noisy_Circuits_Tutorial_with_Purification_Circuits.ipynb).