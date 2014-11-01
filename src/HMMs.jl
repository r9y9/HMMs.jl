module HMMs

# Hidden Markov Models (HMMs)

export HMM, fit!, nstates

for fname in ["hmm"]
    include(string(fname, ".jl"))
end

end # module
