using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(42)

function run_rayleigh_experiment(T::AbstractQuasiNewtonUpdateRule, n::Int)
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
        stopping_criterion = StopWhenAny(
            StopAfterIteration(max(1000)),
            StopWhenGradientNormLess(10^(-6)),
        )
    )
end
io = IOBuffer()

for T in [InverseSR1(), SR1(), BFGS(), InverseBFGS()], n in [100, 300]
    b = @benchmark run_rayleigh_experiment($T, $n) samples=10 seconds=60
    show(io, "text/plain", b)
    s = String(take!(io))
    println("Benchmarking $(n), $(T):\n", s, "\n\n")
end

