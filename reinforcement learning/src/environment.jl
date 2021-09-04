using QuantumClifford
using QuantumClifford.Experimental.NoisyCircuits
using Random
using AbstractAlgebra
using Quantikz: displaycircuit
using Statistics
using StatsBase
using DataFrames
using ReinforcementLearning
using Flux
using Flux.Losses: huber_loss
using ClosedIntervals
using Zygote
using ComponentArrays
using StableRNGs
using QuantumClifford.Experimental.NoisyCircuits: applyop!, affectedqubits, applyop_branches
using Plots

include("MyExplorer1.jl")
include("MyHook.jl")
include("CheckOp.jl")
include("helper.jl")

struct CircuitEnvParams
    initial_pairs::Int64
    net_noise::Float64
    local_noise::Float64
    entanglement_type::Symbol
    max_len::Int64
end

function CircuitEnvParams(;
    initial_pairs = 2,
    net_noise = 0.1,
    local_noise = 0.01,
    entanglement_type = :bell,
    max_len = 5
)
    CircuitEnvParams(
        initial_pairs,
        net_noise,
        local_noise,
        entanglement_type,
        max_len
    )
end

mutable struct CircuitEnv{R<:AbstractRNG} <: RLBase.AbstractEnv
    params::CircuitEnvParams
    curr_fidelity
    last_fidelity
    duplicate::Bool
    circuit
    observation # Component Arrays
    state # Vector
    state_space
    action
    op_space
    action_space
    done::Bool
    measured
    trajectories::Int64
    rng::R
end

function CircuitEnv(;
        initial_pairs = 2,
        net_noise = 0.1,
        local_noise = 0.01,
        entanglement_type = :bell,
        max_len = 5,
        trajectories = 500,
        rng = Random.GLOBAL_RNG,
    )
    params = CircuitEnvParams(
        initial_pairs,
        net_noise,
        local_noise,
        entanglement_type,
        max_len
    )
    op_space = actionspace(initial_pairs)
    action_space = Base.OneTo(length(op_space))
    state_space =Space(
        ClosedInterval[0..1 for i in 1:17*initial_pairs],
        )
    observation =ComponentArray(
        fidelities=zeros(Float64,initial_pairs,4), 
        gates=[(perm=zeros(6),control=0,target=zeros(3)) for _ in 1:initial_pairs], 
        measurements=zeros(Float64,initial_pairs,3)
        )
    input_fidelity=calculate_input_fidelity(net_noise, initial_pairs)
    env = CircuitEnv(
        params,
        input_fidelity,
        input_fidelity,
        false,
        [],
        observation,
        collect(observation),
        state_space,
        rand(action_space),
        op_space,
        action_space,
        false,
        trues(initial_pairs),
        trajectories,
        rng
    )
#     reset!(env)
#     env
end

Random.seed!(env::CartPoleEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::CircuitEnv) = env.action_space
RLBase.state_space(env::CircuitEnv) = env.state_space    
    
# traits
RLBase.NumAgentStyle(::CircuitEnv) = SingleAgent()
RLBase.DynamicStyle(::CircuitEnv) = SEQUENTIAL
RLBase.ActionStyle(::CircuitEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::CircuitEnv) = PERFECT_INFORMATION
RLBase.StateStyle(::CircuitEnv) = Observation{Any}()
RLBase.RewardStyle(::CircuitEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::CircuitEnv) = GENERAL_SUM
RLBase.ChanceStyle(::CircuitEnv) = DETERMINISTIC

"""
reward scheme：

for each added action, if length of circuit > threshold, return -1 and then mark env as done
rewards for Cnots and Measurements are changes in fidelity except for repeated operators
rewards for Stop is change in fidelity if the environment is done, else 0

"""

# function RLBase.reward(env::CircuitEnv)
#     length(env.circuit)<=env.params.max_len || return -1
#     return env.done ? env.curr_fidelity-env.last_fidelity : 0
# end

scaling_factor = 5

RLBase.reward(env::CircuitEnv) = reward(env,env.action)

RLBase.reward(env::CircuitEnv,a::Int64) = action_reward(env,env.op_space[a])

function action_reward(env::CircuitEnv,a::Measurement)
#     length(env.circuit)<=env.params.max_len || return -1
    if length(env.circuit)>env.params.max_len
        env.done=true # Qn: will the environment be updated? Is environment taken in as a copy?
        return -1
    end
#     if env.duplicate return -1 end
    return scaling_factor*(env.curr_fidelity-env.last_fidelity)
end
    
function action_reward(env::CircuitEnv,a::Cnot)
#     length(env.circuit)<=env.params.max_len || return -1
    if length(env.circuit)>env.params.max_len
        env.done=true
        return -1
    end
#     if env.duplicate return -1 end
    return scaling_factor*(env.curr_fidelity-env.last_fidelity)
end

function action_reward(env::CircuitEnv,a::Stop)
#     length(env.circuit)<=env.params.max_len || return -1
    if length(env.circuit)>env.params.max_len
        return -1
    end
    return scaling_factor*(env.curr_fidelity-env.last_fidelity) # scale up the rewards
#     return all(env.measured) ? env.curr_fidelity-env.last_fidelity : -1
end

# RLBase.is_terminated(env::CircuitEnv) = env.done || length(env.circuit)>env.params.max_len
RLBase.is_terminated(env::CircuitEnv) = env.done
RLBase.state(env::CircuitEnv)=env.state
    
function RLBase.reset!(env::CircuitEnv)
    input_fidelity = calculate_input_fidelity(env.params.net_noise,env.params.initial_pairs)
    env.circuit = []
    env.curr_fidelity=input_fidelity
    env.last_fidelity=input_fidelity
    env.observation .= 0
    env.state .= 0
    env.measured .= true
    env.done = false
end

function (env::CircuitEnv)(a::Int64)
    @assert a in env.action_space
    env.action = a
    action=env.op_space[a]
    _add!(env, action)
end

function _add!(env::CircuitEnv, operator::Cnot)
    t1,p1,t2,p2=operator.t1,operator.p1,operator.t2,operator.p2
    if t1!=1 env.measured[t1]=false end
    if t2!=1 env.measured[t2]=false end
#     if length(env.circuit)>0 
#         env.duplicate = (operator == env.circuit[end]) ? true : false
#     end
    push!(env.circuit,operator)
    update_observation!(env, t1, t2, p1, 0)
    update_observation!(env, t2, t1, p2, 1)
    iscomplete!(env)
end

function update_observation!(env::CircuitEnv,qubit,target,perm,control)
    env.observation.gates[qubit].perm .=0
    env.observation.gates[qubit].perm[perm]=1
    env.observation.gates[qubit].control=control
    env.observation.gates[qubit].target .= 0
    env.observation.gates[qubit].target[target]=1
end

function _add!(env::CircuitEnv, operator::Measurement)
#     env.duplicate = env.measured[operator.t] ? true : false
    env.measured[operator.t]=true
    push!(env.circuit,operator)
    update_measurement!(env,operator.t,operator.m)
    iscomplete!(env)
end

function _add!(env::CircuitEnv, operator::Stop)
    env.done=true
end  

function update_measurement!(env::CircuitEnv,qubit,measurement)
    env.observation.measurements[qubit,:] .= 0
    env.observation.measurements[qubit,measurement] = 1
end
    
function iscomplete!(env::CircuitEnv)
    update_fidelity(env)
    env.state=collect(env.observation)
    if all(env.measured)
        env.done=true
    else        
        env.done=false end
end

function update_fidelity(env::CircuitEnv)
    N=env.params.initial_pairs
    netnoise=env.params.net_noise
    localnoise=env.params.local_noise
    env.observation.fidelities = calculate_fidelity(convert(env.circuit,netnoise,N),N,netnoise,localnoise;trajectories=env.trajectories)
    env.last_fidelity=env.curr_fidelity
    env.curr_fidelity=ave_fidelity(env)
end

ave_fidelity(env::CircuitEnv)=sum(env.observation.fidelities[:,1])/env.params.initial_pairs

env=CircuitEnv(
        initial_pairs = 3,
        net_noise = 0.1,
        local_noise = 0.01,
        entanglement_type = :bell,
        max_len=6
)


