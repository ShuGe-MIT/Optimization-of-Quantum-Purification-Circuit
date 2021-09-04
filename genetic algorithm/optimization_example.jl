include("genetic_algorithm.jl")
# Assume our genetic algorithm to optimize output fidelity

iterations=5
sample_size=5
N=5
MAX_OPS=10
netnoise_value=0.1
localnoise_value = 0.01
df=optimizer_Fout(iterations,sample_size,N,MAX_OPS,netnoise_value,localnoise_value)

for i in 1:5
    displaycircuit(convert(df[!,:circuit][i],0.1,df[!,:N][i]),mode=:expanded)
end

display_result_Fout(df,netnoise_value,N)

f1=plot(title="Yield vs. Iterations")
plot!(df[!,:itn][1:10],df[!,:hashing_yield][1:10],marker=true,ylim=(0,0.4),xlabel="iteration",ylabel="yield",legend=false)
f2=plot(title="Entropy vs. Iterations")
plot!(df[!,:itn][1:10],df[!,:entropy][1:10],marker=true,ylim=(-0.5,0.6),xlabel="iteration",ylabel="entropy",legend=false)
f3=plot(title="Purified Fidelity vs. Iterations")
plot!(df[!,:itn][1:10],df[!,:Fout][1:10],marker=true,legend=false,ylim=(0,1),xlabel="iteration",ylabel="output fidelity")
f4=plot(title="Success rate vs. Iterations")
plot!(df[!,:itn][1:10],df[!,:success_rate][1:10],marker=true,legend=false,ylim=(0,1),xlabel="iteration",ylabel="success rate")
f5=plot(title="Number of raw pairs used vs. Iterations")
plot!(df[!,:itn][1:10],df[!,:N][1:10],legend=false,marker=true,ylim=(1,4),xlabel="iteration",ylabel="number of raw pairs used")
f6=plot(title="Length of optimal circuits vs. Iterations")
plot!(df[!,:itn][1:10],df[!,:length][1:10],legend=false,ylim=(1,5),marker=true,xlabel="iteration",ylabel="length of optimal circuits")