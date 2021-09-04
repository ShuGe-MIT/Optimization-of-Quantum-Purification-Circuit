include("helper.jl")

"""
A customized hook to record circuit, length of circuit, and fidelity if the circuit is complete (else 0).
"""
Base.@kwdef struct MyHook <: AbstractHook
    full_info::Bool = false
    convert::Bool = false
    circuits::Vector{Any} = []
    converted_circuits::Vector{Any} = []
    circuit_len::Vector{Int64} = []
    fidelity::Vector{Float64} = []
    # collect after each step
    circuit_len_step::Vector{Int64} = []
    circuit_state_step::Vector{Bool} = []
    fidelity_step::Vector{Float64} = []
    finalcircuit::Vector{Any} = []
end

Base.getindex(h::MyHook) = h.circuit_len

function (hook::MyHook)(::PostEpisodeStage, agent, env)
    if hook.full_info
        push!(hook.circuits,env.circuit)
        if hook.convert push!(hook.converted_circuits, convert(env.circuit,env.params.net_noise,env.params.initial_pairs)) end
        push!(hook.circuit_len,length(env.circuit))
    end
    push!(hook.fidelity,ave_fidelity(env))
end

function (hook::MyHook)(::PostActStage, agent, env)
    if hook.full_info
        push!(hook.circuit_state_step,env.done)
        push!(hook.circuit_len_step,length(env.circuit))
        push!(hook.fidelity_step,ave_fidelity(env))
    end
end

function (hook::MyHook)(::PostExperimentStage, agent, env)
    push!(hook.finalcircuit,convert(env.circuit,env.params.net_noise,env.params.initial_pairs))
end