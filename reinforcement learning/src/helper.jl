using DataFrames
"""
# circuit example
gate=(Hadamard*Phase*Hadamard)⊗(Phase*Hadamard*Phase)
circuit = [
    SparseGate(gate,[1,2]),
    SparseGate(gate,[3,4]),
    SparseGate(CNOT, [1,3]),
    SparseGate(CNOT, [2,4]),
    BellMeasurement([X,X], [3,4]),
    CheckOp([1,2,3,4])
]
"""

good_bell_state = S"XX
                    ZZ"
canonicalize_rref!(good_bell_state)[1]

# define Cnot and Measurement structure
struct Cnot
    t1
    t2
    p1
    p2
end

gates(N)=[Cnot(t1,t2,p1,p2) for p1 in 1:6 for p2 in 1:6 for t1 in 1:N for t2 in t1+1:N]

struct Measurement
    t
    m
end

struct Stop
end

measurements(N)=[Measurement(t,m) for m in 1:3 for t in 2:N]

actionspace(N)=Tuple([gates(N)...,measurements(N)...,Stop()])

function state_tensor_pow(state::Stabilizer,power::Integer) # MODIFIED less strict type so it works on all Stabilizers
    result=state
    for i in 1:power-1
        result=result⊗state
    end
    return result
end

function append_permutation_op!(circuit,permutation,qubit)
    if permutation==2
        gate=Hadamard⊗Hadamard
    elseif permutation==3
        gate=(Hadamard*Phase*Hadamard)⊗(Phase*Hadamard*Phase)
    elseif permutation==4
        gate=(Phase*Hadamard)⊗(Hadamard*Phase*Hadamard*Phase)
    elseif permutation==5
        gate=(Phase*Hadamard*Phase*Hadamard)⊗(Hadamard*Phase*Hadamard*Phase*Hadamard*Phase*Hadamard*Phase)
    elseif permutation==6
        gate=(Hadamard*Phase*Hadamard*Phase*Hadamard)⊗(Hadamard*Hadamard*Phase*Hadamard*Phase*Hadamard*Phase*Hadamard*Phase)
    end
    if permutation!=1
        push!(circuit,SparseGate(gate,[qubit*2-1,qubit*2]))
    end
end 

function append_op!(circuit,operation::Cnot,netnoise)
    append_permutation_op!(circuit,operation.p1,operation.t1)
    append_permutation_op!(circuit,operation.p2,operation.t2)
    Base.append!(circuit,[SparseGate(CNOT,[operation.t1*2-1,operation.t2*2-1]),SparseGate(CNOT,[operation.t1*2,operation.t2*2])])
end

function append_op!(circuit,operation::Measurement,netnoise)
    if operation.m==1
        m1=m2=X
    elseif operation.m==2
        m1=Y
        m2=-Y
    else
        m1=m2=Z
    end
    push!(circuit,BellMeasurement([m1,m2],[operation.t*2-1,operation.t*2]))
    push!(circuit,Reset(good_bell_state,[operation.t*2-1,operation.t*2]))
    push!(circuit,NoiseOp(netnoise,[operation.t*2-1,operation.t*2]))
end

function convert(abstract_circuit,netnoise_value::Float64,N::Int64)
    circuit=[]
    netnoise = UnbiasedUncorrelatedNoise(netnoise_value/N)
    for op in abstract_circuit
        append_op!(circuit,op,netnoise)
    end
    push!(circuit,CheckOp([i for i in 1:N*2]))
    return circuit
end

make_noisy(g::SparseGate, noise) = NoisyGate(g, UnbiasedUncorrelatedNoise(1//3*noise))
make_noisy(m::BellMeasurement, noise) = NoisyBellMeasurement(m, noise)
make_noisy(other_op, noise) = other_op
make_noisy(circuit::AbstractVector, noise) = [make_noisy(op, noise) for op in circuit];

function calculate_input_fidelity(netnoise_value,N)
    nopurification_circuit = [VerifyOp(good_bell_state, [1,2])]
    netnoise = UnbiasedUncorrelatedNoise(netnoise_value/N)
    netnoise_opall = NoiseOpAll(netnoise);
    netnoise_nopurification = petrajectories(good_bell_state,
                                            [netnoise_opall,nopurification_circuit...],max_order=2)
    F0=round(netnoise_nopurification[:true_success],digits=3)
    return F0
end

mutable struct TrajectoriesOverStep
    trajectories_stable::Int64
    trajectories_init::Int64
    warmup_steps::Int
    increase_steps::Int
    curr_step::Int
    is_training::Bool
end

function TrajectoriesOverStep(
        increase_steps;
        warmup_steps = 100,
        curr_step = 1, 
        trajectories_stable = 1000, 
        trajectories_init = 10, 
        is_training = true
    )
    TrajectoriesOverStep(
        trajectories_stable, 
        trajectories_init, 
        warmup_steps, 
        increase_steps, 
        curr_step, 
        is_training
    )
end

function (s::TrajectoriesOverStep)(args...)
    s.is_training && (s.curr_step += 1)
end

function get_trajectories(s::TrajectoriesOverStep, step)
    if step <= s.warmup_steps
        s.trajectories_init
    elseif step >= (s.warmup_steps + s.increase_steps)
        s.trajectories_stable
    else
        s.trajectories_init + floor(Int,(step - s.warmup_steps) / s.increase_steps * (s.trajectories_stable - s.trajectories_init))
    end
end

get_trajectories(s::TrajectoriesOverStep) = s.is_training ? get_trajectories(s, s.curr_step) : 1000

function calculate_fidelity(s::TrajectoriesOverStep,circuit,N,netnoise_value,localnoise_value)
    initial_state=state_tensor_pow(good_bell_state,N)
    netnoise = UnbiasedUncorrelatedNoise(netnoise_value/N)
    netnoise_opall = NoiseOpAll(netnoise);
    c = [netnoise_opall,circuit...]
    c = make_noisy(c, localnoise_value)
    trajectories = get_trajectories(s)
    return mymctrajectories(initial_state, c, N, trajectories = trajectories)[2]
end

function mymctrajectories(initialstate,circuit,initial_pairs;trajectories=500)
    states=[mctrajectory!(copy(initialstate),circuit)[2] for i in 1:trajectories]
    success_rate=1-(count(s->s==:detected_failure,states)/trajectories)
    states = filter(s->s!=:detected_failure,states)
    if length(states)>0
        states=hcat(states...)
        result=[merge(Dict([(k=>0) for k in [:A,:B,:C,:D]]),proportionmap(states[i,:])) for i in 1:size(states)[1]]
        return success_rate, Tables.matrix(select!(DataFrame(result),[:A,:B,:C,:D]))
    else
        return success_rate, zeros(Float64,initial_pairs,4)
    end
end