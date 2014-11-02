using Distributions

# todo: proper parent type
abstract HiddenMarkovModel{VF,VS,Component<:Distribution} <: Distribution{VF,VS}

# Note that distribution C must implement fit_mle, maximum likelihood estimation.
type HMM{VF,VS,C<:Distribution} <: HiddenMarkovModel{VF,VS,C}
    π::Array{Float64, 1}  # initial state prob., shape (K,)
    A::Array{Float64, 2}  # transition matrix, shape (K, K)
    B::Vector{C}          # vector of observation model, shape (K,)

    function HMM(p::Vector{Float64}, A::Matrix{Float64}, B::Vector{C})
        if length(p) != length(B)
            throw(DimentionMismatch("Inconsistent dimentions"))
        end
        if !isprobvec(p)
            throw(ArgumentError("p = $p is not a probability vector."))
        end
        new(p, A, B)
    end
end

function HMM{C<:Distribution}(p::Vector{Float64},
                              A::Matrix{Float64},
                              B::Vector{C})
    VF = Distributions.variate_form(C)
    VS = Distributions.value_support(C)
    HMM{VF,VS,C}(p, A, B)
end

function HMM{C<:Distribution}(B::Vector{C})
    K = length(B)
    HMM(ones(K)/K, ones(K, K)/K, B)
end

nstates(hmm::HMM) = length(hmm.B)
Base.length(hmm::HMM) = length(hmm.B[1])
Base.size(hmm::HMM) = (length(hmm), 1) # length for one sample

# E-step
function updateE!{F,S,C<:Distribution}(hmm::HMM{F,S,C},
                                       Y::AbstractMatrix,    # shape: (D, T)
                                       α::Matrix{Float64},   # shape: (K, T)
                                       β::Matrix{Float64},   # shape: (K, T)
                                       γ::Matrix{Float64},   # shape: (K, T)
                                       ξ::Array{Float64, 3}, # shape: (K, K, T-1)
                                       B::Matrix{Float64})   # shape: (K, T)
    const D, T = size(Y)
    
    # scaling paramter
    c = Array(Float64, T)

    # forward recursion
    α[:,1] = hmm.π .* B[:,1]
    α[:,1] /= (c[1] = sum(α[:,1]))
    for t=2:T
        @inbounds α[:,t] = (hmm.A' * α[:,t-1]) .* B[:,t]
        @inbounds α[:,t] /= (c[t] = sum(α[:,t]))
    end
    @assert !any(isnan(α))

    likelihood = sum(log(c))

    # backword recursion
    β[:,T] = 1.0
    for t=T-1:-1:1
        @inbounds β[:,t] = hmm.A * β[:,t+1] .* B[:,t+1] ./ c[t+1]
    end
    @assert !any(isnan(β))

    γ = α .* β

    for t=1:T-1
        @inbounds ξ[:,:,t] = hmm.A .* α[:,t] .* β[:,t+1]' .* B[:,t+1]' ./ c[t+1]
    end

    return γ, ξ, likelihood
end

# M-step
function updateM!{F,S,C<:Distribution}(hmm::HMM{F,S,C},
                                       Y::AbstractMatrix,    # shape: (D, T)
                                       γ::Matrix{Float64},   # shape: (K, T)
                                       ξ::Array{Float64, 3}) # shape: (K, K, T-1)
    const D, T = size(Y)
    const K = length(hmm.B)

    # prior
    hmm.π[:] = γ[:,1] / sum(γ[:,1])

    # transition
    hmm.A[:,:] = sum(ξ, 3) ./ sum(γ[:,1:end-1], 2)

    # ensure sum of A[:,i] = 1.0
    hmm.A[:,:] ./= sum(hmm.A, 2)

    # observation
    for k=1:K
        hmm.B[k] = fit_mle(C, Y, γ[k,:])
    end
end

type HMMTrainingResult
    likelihoods::Vector{Float64}
    α::Matrix{Float64}
    β::Matrix{Float64}
    γ::Matrix{Float64}
    ξ::Array{Float64, 3}
end

function compute_observation_prob!(hmm::HMM,
                                   Y::AbstractMatrix,
                                   B::Matrix{Float64})
    K = nstates(hmm)
    for k=1:K
        for t=1:size(Y,2)
            @inbounds B[k,t] = pdf(hmm.B[k], Y[:,t])
        end
    end
    B
end

function fit!{F,S,C<:Distribution}(hmm::HMM{F,S,C},
                                   Y::AbstractMatrix;
                                   maxiter::Int=100,
                                   tol::Float64=-1.0,
                                   verbose::Bool=false)
    const D, T = size(Y)
    const K = length(hmm.B)
    
    likelihood::Vector{Float64} = zeros(1)

    α = Array(Float64, K, T)
    β = Array(Float64, K, T)
    γ = Array(Float64, K, T)
    ξ = Array(Float64, K, K, T-1)
    B = Array(Float64, K, T)

    # Roop of EM-algorithm
    for iter=1:maxiter
        B = compute_observation_prob!(hmm, Y, B)

        # update expectations
        γ, ξ, score = updateE!(hmm, Y, α, β, γ, ξ, B)

        # update parameters
        updateM!(hmm, Y, γ, ξ)

        improvement = (score - likelihood[end]) / abs(likelihood[end])

        if verbose
            println("#$(iter): bound $(likelihood[end])
                    improvement: $(improvement)")
        end
        
        # check if converged
        if abs(improvement) < tol
            if verbose
                println("#$(iter) converged")
            end
            break
        end

        push!(likelihood, score)
    end
    
    # remove initial zero
    shift!(likelihood)
    
    return HMMTrainingResult(likelihood, α, β, γ, ξ)
end

function decode{F,S,C<:Distribution}(hmm::HMM{F,S,C},
                                     Y::AbstractMatrix)
    const D, T = size(Y)
    D == length(hmm) || throw(DimentionMismatch("Inconsistent dimentions"))

    const K = nstates(hmm)
    ψ = zeros(Float64, K, T)
    Ψ = zeros(Int, K, T)

    logπ = log(hmm.π)
    logA = log(hmm.A)
    for k=1:K
        @inbounds ψ[k,1] = logπ[k] + logpdf(hmm.B[k], Y[:,1])
    end
    Ψ[:,1] = indmax(ψ[:,1])

    # forward recursion
    for t=2:T
        for k=1:K
            @inbounds p = ψ[:,t-1] + hmm.A[:,k] + logpdf(hmm.B[k], Y[:,t])
            ψ[k,t] = maximum(p)
            Ψ[k,t] = indmax(p)
        end
    end

    # backtrace
    z = zeros(Int, T)
    z[end] = indmax(ψ[:,end])
    for t=T-1:-1:1
        @inbounds z[t] = Ψ[z[t+1],t]
    end

    return z
end
