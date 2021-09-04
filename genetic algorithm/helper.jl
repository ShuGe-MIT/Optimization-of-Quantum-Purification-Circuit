using QuantumClifford
using QuantumClifford.Experimental.NoisyCircuits
using Plots
using Random
using AbstractAlgebra
using LaTeXStrings
using Quantikz: displaycircuit
using Statistics
using StatsBase
using DataFrames

good_bell_state = S"XX
                    ZZ"
canonicalize_rref!(good_bell_state)[1]

function state_tensor_pow(state::Stabilizer{Vector{UInt8}, Matrix{UInt64}},power::Integer)
    result=state
    for i in 1:power-1
        result=result⊗state
    end
    return result
end

# generate random circuit

function add_random_cnot!(circuit,qList,N)
    result=0
    while result==0
        pair=sample(1:N,2,replace=false) #preserved qubits can never be the target qubits
        if (qList[pair[1]] && pair[2]!=1)
            circuit=append!(circuit,[("CN",[rand(1:6),rand(1:6)],pair)])
            qList[pair[2]]=true
            result=1
        end
    end
    return circuit,qList
end

function add_random_measurement!(circuit,qList,N)
    #probability of performing measurement on trash qubits    
    #equal opportunities of measurement for coinX, coinZ, or antiY
    #select a random number of qubits to apply measurements
    qubit_=sample(2:N,rand(1:N-1),replace=false)
    qubit=[]
    for q in qubit_
        if (qList[q])
            qubit=append!(qubit,[q])
        end
    end
    if length(qubit)>0
        circuit=append!(circuit,[("M",[rand(1:3)],qubit)])
        qList[qubit].=false
    end
    return circuit,qList
end

function add_random_operation!(circuit,qList,N)
    #pick a random pair from currently available qubits
    rand1=rand()
    if 0.33<rand1<0.66 #cnot_gate
        circuit,qList=add_random_cnot!(circuit,qList,N)
    elseif rand1>=0.66
        circuit,qList=add_random_measurement!(circuit,qList,N)
    end
    return circuit,qList
end

function generate_abstract_circuit(N,MAX_OPS)
    qList=falses(1,N) #qList keeps track of whether the qubit is worth measuring
    qList[1]=true 
    circuit=[]
    while length(circuit)==0
        for opt in 1:MAX_OPS
            #equal probabilities of applying nothing, CPHASE, CNOT, measurement (coinX, coinZ, antiY) (what are the permitted gates)
            circuit,qList=add_random_operation!(circuit,qList,N)
        end
    end
    circuit_=[circuit[1]]
    for i in 2:length(circuit)
        if circuit[i-1]!=circuit[i]
            circuit_=append!(circuit_,[circuit[i]])
        end
    end
    # Add measurement to all trash qubits that are not measured
    unmeasured=[]
    for i in 2:N
        if qList[i]
            push!(unmeasured,i)
        end
    end
    if length(unmeasured)>0
        append!(circuit_,[("M",[rand(1:3)],unmeasured)])
    end
    return circuit_,qList
end

# convert abstract circuit to circuit in QuantumClifford.jl

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
        circuit=append!(circuit,[SparseGate(gate,[qubit*2-1,qubit*2])])
    end
    return circuit
end 

function append_permutation_cnot!(circuit,permutation,pair)
    a,b=pair
    circuit=append_permutation_op!(circuit,permutation[1],a)
    circuit=append_permutation_op!(circuit,permutation[2],b)
    circuit=append!(circuit,[SparseGate(CNOT,[a*2-1,b*2-1]),SparseGate(CNOT,[a*2,b*2])])
    return circuit
end

function convert(abstract_circuit,netnoise_value,N)
    #convert abstract circuit representation to complete representation
    circuit=[]
    netnoise = UnbiasedUncorrelatedNoise(netnoise_value/N)
    for i in abstract_circuit
        if i[1]=="M"
            if i[2][1]==1
                measurement1=X
                measurement2=X
            elseif i[1][1]==2
                measurement1=Y
                measurement2=-Y
            else
                measurement1=Z
                measurement2=Z
            end
            for q in i[3]
                circuit=append!(circuit,[BellMeasurement([measurement1,measurement2],[q*2-1,q*2])])
                circuit=append!(circuit,[Reset(good_bell_state,[q*2-1,q*2])])
                circuit=append!(circuit,[NoiseOp(netnoise,[q*2-1,q*2])])
            end
        else
            circuit=append_permutation_cnot!(circuit,i[2],i[3])
        end
    end
    circuit=append!(circuit,[VerifyOp(good_bell_state, [1,2])])
    return circuit
end

function get_qList(abstract_circuit,N)
    qList=falses(1,N)
    qList[1]=true
    for i in abstract_circuit
        if i[1]=="M"
            for q in i[3]
                qList[q]=false
            end
        else
            qList[i[3]].=true
        end
    end
    return qList
end

function generate_random_circuit(N,MAX_OPS,netnoise_value)
    # currently not used in the optimizer
    circuit,qList=generate_abstract_circuit(N,MAX_OPS)
    circuit_=convert(circuit,netnoise_value,N)
    return circuit_
end