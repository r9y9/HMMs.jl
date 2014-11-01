module HMMs

# Hidden Markov Models (HMMs)

export HMM, fit!

for fname in ["hmm"]
    include(string(fname, ".jl"))
end

end # module
