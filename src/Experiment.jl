using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(42)

function run_rayleigh_experiment(T::AbstractQuasiNewtonUpdateRule, S::Stepsize, n::Int)
    A = randn(n, n)
    A = (A + A') / 2
    F(X::Array{Float64,1}) = X' * A * X
    ∇F(X::Array{Float64,1}) = 2 * (A * X - X * X' * A * X)
    M = Sphere(n - 1)
    x = random_point(M)
    return quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size = -1,
        direction_update = T,
        step_size = S,
        stopping_criterion = StopWhenAny(
            StopAfterIteration(max(1000)),
            StopWhenGradientNormLess(10^(-6)),
        )
    )
end
io = IOBuffer()

for T in [BFGS(), InverseBFGS(), SR1(), InverseSR1(), SR1(10^(-8)), InverseSR1(10^(-8))], n in [100, 300]
    b = @benchmark run_rayleigh_experiment($T, $(WolfePowellLineseach(ExponentialRetraction(), ParallelTransport())), $n) samples = 10
    show(io, "text/plain", b)
    s = String(take!(io))
    println("Benchmarking $(n), $(T):\n", s, "\n\n")
end

