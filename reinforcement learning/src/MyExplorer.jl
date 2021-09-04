using Random
using Distributions: Categorical
using Flux

mutable struct MyExplorer{Kind,IsBreakTie,R} <: AbstractExplorer
    ϵ_stable::Float64
    ϵ_init::Float64
    warmup_steps::Int
    decay_steps::Int
    step::Int
    rng::R
    is_training::Bool
end

function MyExplorer(;
    ϵ_stable,
    kind = :linear,
    ϵ_init = 1.0,
    warmup_steps = 0,
    decay_steps = 0,
    step = 1,
    is_break_tie = false,
    is_training = true,
    rng = Random.GLOBAL_RNG,
)
    MyExplorer{kind,is_break_tie,typeof(rng)}(
        ϵ_stable,
        ϵ_init,
        warmup_steps,
        decay_steps,
        step,
        rng,
        is_training,
    )
end

MyExplorer(ϵ; kwargs...) = MyExplorer(; ϵ_stable = ϵ, kwargs...)

function get_ϵ(s::MyExplorer{:linear}, step)
    if step <= s.warmup_steps
        s.ϵ_init
    elseif step >= (s.warmup_steps + s.decay_steps)
        s.ϵ_stable
    else
        steps_left = s.warmup_steps + s.decay_steps - step
        s.ϵ_stable + steps_left / s.decay_steps * (s.ϵ_init - s.ϵ_stable)
    end
end

function get_ϵ(s::MyExplorer{:exp}, step)
    if step <= s.warmup_steps
        s.ϵ_init
    else
        n = step - s.warmup_steps
        scale = s.ϵ_init - s.ϵ_stable
        s.ϵ_stable + scale * exp(-1.0 * n / s.decay_steps)
    end
end

get_ϵ(s::MyExplorer) = s.is_training ? get_ϵ(s, s.step) : 0.0

function (s::MyExplorer{<:Any,false})(values)
    ϵ = get_ϵ(s)
    s.is_training && (s.step += 1)
    rand(s.rng) >= ϵ ? findmax(values)[2] : rand(s.rng, 1:length(values))
end


function (s::MyExplorer{<:Any,false})(values, mask)
    ϵ = get_ϵ(s)
    s.is_training && (s.step += 1)
    rand(s.rng) >= ϵ ? findmax(values, mask)[2] : rand(s.rng, findall(mask))
end

Random.seed!(s::MyExplorer, seed) = Random.seed!(s.rng, seed)

function RLBase.prob(s::MyExplorer{<:Any,false}, values)
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    probs[findmax(values)[2]] += 1 - ϵ
    Categorical(probs)
end

function RLBase.prob(s::MyExplorer{<:Any,false}, values, action::Integer)
    ϵ, n = get_ϵ(s), length(values)
    if action == findmax(values)[2]
        ϵ / n + 1 - ϵ
    elseif action > n-6 # TODO: currently hard-coded (6 possible measurements), need to add another argument for initial number of pairs
        ϵ / 3 / 6
    else
        2 * ϵ / 3 / (n-6)
    end
end


function RLBase.prob(s::MyExplorer{<:Any,false}, values, mask)
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    probs[findmax(values, mask)[2]] += 1 - ϵ
    Categorical(probs)
end