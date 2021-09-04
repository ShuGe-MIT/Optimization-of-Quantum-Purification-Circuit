using QuantumClifford: AbstractStabilizer
import QuantumClifford.Experimental.NoisyCircuits: applyop!, affectedqubits, applyop_branches

struct CheckOp <: AbstractOperation
    indices::AbstractVector{Int}
    CheckOp(indices) = new(indices)
end

"""
A=S"ZZ
XX"

B=S"-ZZ
-XX"

C=S"-ZZ
XX"

D=S"ZZ
-XX"
"""

function applyop!(s::AbstractStabilizer, v::CheckOp)
    s, _ = canonicalize_rref!(s,v.indices)
    sv = stabilizerview(s)
    final_states=[]
    for i in (size(sv)[1]÷2):-1:1 #reverse order
        if sv.phases[i*2-1]==0 && sv.phases[i*2]==0 push!(final_states, :A)
        elseif sv.phases[i*2-1]!=0 && sv.phases[i*2]!=0 push!(final_states, :B)
        elseif sv.phases[i*2-1]!=0 && sv.phases[i*2]==0 push!(final_states, :C)
        else push!(final_states, :D)
        end
    end
    return s, final_states
end

affectedqubits(v::CheckOp) = v.indices
applyop_branches(s::AbstractStabilizer, v::CheckOp; max_order=1) = [(applyop!(copy(s),v)...,1,0)] 