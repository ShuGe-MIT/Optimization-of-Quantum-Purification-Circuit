include("helper.jl")
# Optimization process

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

function generate_child_pairs(available_circuits,sample_size,N,MAX_OPS)
    circuits=[]
    for i in 1:sample_size
        if rand()<0.5
            circuit1=available_circuits[rand(1:length(available_circuits))]
        else
            circuit1,qList=generate_abstract_circuit(N,MAX_OPS)
        end
        if rand()<0.5
            circuit2=available_circuits[rand(1:length(available_circuits))]
        else
            circuit2,qList=generate_abstract_circuit(N,MAX_OPS)
        end
        if length(circuit1)>0
            new_circuit=circuit1[1:rand(1:length(circuit1))]
        else
            new_circuit=circuit1
        end
        if length(circuit2)>0
            append!(new_circuit,circuit2[rand(1:length(circuit2)):length(circuit2)])
        end
        if new_circuit[end][1]=="M"
            pop!(new_circuit)
        end
        qList=get_qList(new_circuit,N)
        unmeasured=[]
        for i in 2:N
            if qList[i]
                push!(unmeasured,i)
            end
        end
        if length(unmeasured)>0
            append!(new_circuit,[("M",[rand(1:3)],unmeasured)])
        end
        append!(circuits,[new_circuit])
    end
    return circuits
end

function new_drop_op(circuit)
    new_circuit=circuit[:]
    r=rand(2:length(circuit)-1)
    if circuit[r-1]!=circuit[r+1] && circuit[r][1]=="CN"
        deleteat!(new_circuit,r)
    end
    return new_circuit
end

function new_add_op(circuit,N)
    if length(circuit)==0
        new_circuit=[]
        new_circuit,_=add_random_cnot!(new_circuit,qList,N)
    else
        m=rand(1:length(circuit))
        new_circuit=circuit[1:m]
        qList=get_qList(new_circuit,N)
        new_circuit,qList=add_random_cnot!(new_circuit,qList,N)
        if m<length(circuit)
            if circuit[m+1]==new_circuit[length(new_circuit)]
                append!(new_circuit,circuit[m+2:length(circuit)])
            else
                append!(new_circuit,circuit[m+1:length(circuit)])
            end
        end
    end
    qList=get_qList(new_circuit,N)
    unmeasured=[]
    for i in 2:N
        if qList[i]
            push!(unmeasured,i)
        end
    end
    if length(unmeasured)>0
        append!(new_circuit,[("M",[rand(1:3)],unmeasured)])
    end
    return new_circuit
end

function new_mutate(circuit,N)
    new_circuit=circuit[:]
    m=rand(1:(length(circuit)))
    if circuit[m][1]=="CN"
        new_circuit[m][2].=[rand(1:6),rand(1:6)]
    end
    qList=get_qList(new_circuit,N)
    unmeasured=[]
    for i in 2:N
        if qList[i]
            push!(unmeasured,i)
        end
    end
    if length(unmeasured)>0
        append!(new_circuit,[("M",[rand(1:3)],unmeasured)])
    end
    return new_circuit
end

function generate_next_generation(available_circuits,sample_size,N,MAX_OPS)
    new_circuits=available_circuits
    #cross-overs
    append!(new_circuits,generate_child_pairs(available_circuits,sample_size,N,MAX_OPS))
    #mutations
    mutated_circuits=[]
    for c in new_circuits
        if (rand()<0.3 && length(c)>2)
            new_circuit=new_drop_op(c)
            push!(mutated_circuits,new_circuit)
        end
        if (rand()<0.3 && length(c)<MAX_OPS)
            new_circuit=new_add_op(c,N)
            push!(mutated_circuits,new_circuit)
        end
        if (rand()<0.3 && length(c)>0)
            new_circuit=new_mutate(c,N)
            push!(mutated_circuits,new_circuit)
        end
    end
    append!(new_circuits,mutated_circuits)
    return new_circuits
end

function calculate_hashing_yield(circuit,N,n,netnoise_value,localnoise_value)
    eps=10^(-10)
    initial_state=state_tensor_pow(good_bell_state,N)
    netnoise = UnbiasedUncorrelatedNoise(netnoise_value/N)
    netnoise_opall = NoiseOpAll(netnoise);
    c = [netnoise_opall,circuit...]
    c = make_noisy(c, localnoise_value)
    pe_allnoise = petrajectories(initial_state, c) #takes longest time
    success_rate=pe_allnoise[:true_success]+pe_allnoise[:undetected_failure]
    if success_rate==0
        return 0,NaN,0,0
    else
        Fout=pe_allnoise[:true_success]/(pe_allnoise[:true_success]+pe_allnoise[:undetected_failure])
        e=-(Fout*log(Fout+eps)+(1-Fout)*log((1-Fout+eps)/3))
        hy=success_rate/n*(1-e)
        return hy,e,Fout,success_rate
    end
end

function calculate_N(circuit)
    n=1
    for i in circuit
        if i[1] in ["M"]
            n+=length(i[2])
        end
    end
    return n
end

function get_Fout_hy(itn,next_available_circuits,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)
    if itn==1
        result=[]
        for i in 1:sample_size
            circuit,qList=generate_abstract_circuit(N,MAX_OPS)
            circuit_=convert(circuit,netnoise_value,N)
            n=calculate_N(circuit)
            hy,e,Fout,success_rate=calculate_hashing_yield(circuit_,N,n,netnoise_value,localnoise_value)
            push!(result,(hashing_yield=hy,entropy=e,Fout=Fout,success_rate=success_rate,N=n,circuits=circuit))
        end
    else
#         println(next_available_circuits)
        circuits=generate_next_generation(next_available_circuits,sample_size,N,MAX_OPS)
#         println(circuits)
        result=[]
        println("sample size = $(length(circuits))")
#         @time begin
        for i in 1:length(circuits)
            circuit_=convert(circuits[i],netnoise_value,N)
            n=calculate_N(circuits[i])
            hy,e,Fout,success_rate=calculate_hashing_yield(circuit_,N,n,netnoise_value,localnoise_value)
            push!(result,(hashing_yield=hy,entropy=e,Fout=Fout,success_rate=success_rate,N=n,circuits=circuits[i]))
        end
#         end
    end
    sort!(result,by=x->x[:hashing_yield])
    result=result[(length(result)-sample_size+1):end]
    df=DataFrame(result)
    return df
end

function get_Fout(itn,next_available_circuits,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)
    if itn==1
        result=[]
        for i in 1:sample_size
            circuit,qList=generate_abstract_circuit(N,MAX_OPS)
            circuit_=convert(circuit,netnoise_value,N)
            n=calculate_N(circuit)
            hy,e,Fout,success_rate=calculate_hashing_yield(circuit_,N,n,netnoise_value,localnoise_value)
            push!(result,(hashing_yield=hy,entropy=e,Fout=Fout,success_rate=success_rate,N=n,circuits=circuit))
        end
    else
        @time circuits=generate_next_generation(next_available_circuits,sample_size,N,MAX_OPS)
        result=[]
        println("sample size = $(length(circuits))")
#         @time begin
        for i in 1:length(circuits)
            qList=get_qList(circuits[i],N)
            circuit_=convert(circuits[i],netnoise_value,N)
            n=calculate_N(circuits[i])
            hy,e,Fout,success_rate=calculate_hashing_yield(circuit_,N,n,netnoise_value,localnoise_value)
            push!(result,(hashing_yield=hy,entropy=e,Fout=Fout,success_rate=success_rate,N=n,circuits=circuits[i]))
        end
#         end
    end
    sort!(result,by=x->x[:Fout])
    result=result[(length(result)-sample_size+1):end]
    df=DataFrame(result)
    return df
end

function optimizer_Fout(iterations,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)
    result=[]
    next_available_circuits=[]
    F0=calculate_input_fidelity(netnoise_value,N)
    for itn in 1:iterations
        println("iteration=$(itn)")
        df=get_Fout(itn,next_available_circuits,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)
#         println(df[!,:circuits])
#         println(df)
        push!(result,(itn=itn,hashing_yield=df[!,:hashing_yield][end],entropy=df[!,:entropy][end],Fout=df[!,:Fout][end],success_rate=df[!,:success_rate][end],circuit=df[!,:circuits][end],N=df[!,:N][end],length=length(df[!,:circuits][end])))
        println(result[itn])
        next_available_circuits=df[!,:circuits][sample_size÷2:end]
    end
    df=DataFrame(result)
    return df
end

function optimizer_hy(iterations,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)
    result=[]
    next_available_circuits=[]
    F0=calculate_input_fidelity(netnoise_value,N)
    for itn in 1:iterations
        println("iteration=$(itn)")
        df=get_Fout_hy(itn,next_available_circuits,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)
        push!(result,(itn=itn,hashing_yield=df[!,:hashing_yield][end],entropy=df[!,:entropy][end],Fout=df[!,:Fout][end],success_rate=df[!,:success_rate][end],circuit=df[!,:circuits][end],N=df[!,:N][end],length=length(df[!,:circuits][end])))
        println(result[itn])
        next_available_circuits=df[!,:circuits][sample_size÷2:end]
    end
    df=DataFrame(result)
    return df
end