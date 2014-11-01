using HMMs
using Distributions
using Base.Test

# Continuous Gaussian HMM
srand(98765)

T = 100
D = 10
g1 = MvNormal(3*ones(D), diagm(ones(D)))
g2 = MvNormal(8*ones(D), diagm(3*ones(D)))

y1 = rand(g1, T)
y2 = rand(g2, T)
y1[:,1:T/2] = 0.0
y2[:,T/2+1:end] = 0.0
Y = y1 + y2

# 2 state continuous HMM
chmm = HMM([g1, g2])
r = fit!(chmm, Y, maxiter=30, verbose=false, tol=-1.0)

# TODO add more tests
@test mean(g1.μ-chmm.B[1].μ) < 0.1
@test mean(g2.μ-chmm.B[2].μ) < 0.1

# Discrete Multinomial HMM
srand(98765)

T = 100
D = 10
m1 = Multinomial(D, ones(D)/D)
m2 = Multinomial(D, (v = rand(D))/sum(v))

y1 = rand(m1, T)
y2 = rand(m2, T)
y1[:,1:T/2] = 0.0
y2[:,T/2+1:end] = 0.0
Y = y1 + y2

# 2 state discrete HMM
dhmm = HMM([m1, m2])
r = fit!(dhmm, Y, maxiter=30, verbose=false, tol=-1.0)

# TODO test
