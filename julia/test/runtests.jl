using Test, CorrelatedHOPS, ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles
BLAS.set_num_threads(1)
function fixture(name)
    return readdlm(joinpath(@__DIR__, "fixtures", name * "_real.csv"), ',', Float64) +
           im * readdlm(joinpath(@__DIR__, "fixtures", name * "_imag.csv"), ',', Float64)
end

@testset "Local representation and Python generator" begin
    e = Engine(
        2; J=0.4, delta=1.3, g=0.3, kappa=[0.1, 0.4], kmax=[1, 2], cutoff=1e-13, maxdim=64
    )
    @test dim.(e.sites) == [2, 2, 2, 3]
    shift = ComplexF64[0.01 + 0.02im, -0.02 + 0.03im]
    z = ComplexF64[0.2 + 0.1im, -0.3 + 0.2im]
    beta = ComplexF64[0.02 + 0.01im, 0.01 - 0.02im]
    mpo = generator(e, shift, z, beta)
    t = reduce(*, mpo)
    # Python C-order basis: spin1, aux1, spin2, aux2 (last index fastest).
    labels = [(s1, k1, s2, k2) for s1 in 1:2 for k1 in 1:2 for s2 in 1:2 for k2 in 1:3]
    mat = zeros(ComplexF64, length(labels), length(labels))
    for (r, rr) in enumerate(labels), (c, cc) in enumerate(labels)
        inds = [prime(e.sites[i]) => rr[i] for i in 1:4]
        append!(inds, [dag(e.sites[i]) => cc[i] for i in 1:4])
        mat[r, c] = t[inds...]
    end
    @test maximum(abs.(mat - fixture("generator"))) < 1e-12
    psi = initial_state(e; preparation=:neel)
    root, n2 = root_mps(e, psi)
    @test n2 ≈ 1
    @test expect(root, "Sz") ≈ [0.5, -0.5]
end

@testset "Compression history survives sparse observations" begin
    e = Engine(2; J=0.8, g=0.7, kmax=2, cutoff=1e-8, maxdim=2)
    dt = 0.02
    steps = 8
    z = stationary_noise(e, dt / 2, 2steps, 31)
    psi = initial_state(e; preparation=:neel)
    original = deepcopy(psi)
    shift = zeros(ComplexF64, 2)

    # Measure predictor truncation independently. It must be represented in
    # advance's returned compression history, even though it is not the output.
    predictor_weights = Float64[]
    predictor_callback(; link, truncation_error) =
        push!(predictor_weights, truncation_error)
    predictor = apply(
        generator(e, shift, z[:, 1], means(e, psi); identity=1.0, scale=dt / 2),
        psi;
        cutoff=e.cutoff,
        maxdim=e.maxdim,
        alg="naive",
        callback=predictor_callback,
    )
    _, _, first_weights = advance(e, psi, shift, z[:, 1], z[:, 2], dt)
    @test maximum(predictor_weights) > 0
    @test first_weights[1:length(predictor_weights)] ≈ predictor_weights
    @test inner(psi - original, psi - original) < 1e-12
    @test shift == zeros(ComplexF64, 2)

    all_weights = Float64[]
    for i in 1:steps
        psi, shift, weights = advance(e, psi, shift, z[:, 2i - 1], z[:, 2i], dt)
        append!(all_weights, weights)
    end
    dense = simulate(e; T=steps * dt, dt, seed=31, observe_every=1)
    sparse = simulate(e; T=steps * dt, dt, seed=31, observe_every=5)
    final = sparse.records[end].diagnostics
    @test length(sparse.records) == 3
    @test final.largest_discarded_weight ≈ maximum(all_weights)
    @test final.largest_discarded_weight > 0
    @test final.bond_cap_reached
    @test final.maximum_bond_dimension == e.maxdim
    @test final == dense.records[end].diagnostics
    @test issorted([r.diagnostics.largest_discarded_weight for r in dense.records])
    @test issorted([r.diagnostics.maximum_bond_dimension for r in dense.records])
end

@testset "Public grid and parameter validation" begin
    e = Engine(1; kmax=1)
    for value in (1.0, true, 0, -1)
        @test_throws ArgumentError Engine(1; kmax=value)
        @test_throws ArgumentError Engine(1; maxdim=value)
        @test_throws ArgumentError simulate(e; T=0.02, dt=0.01, observe_every=value)
    end
    @test_throws ArgumentError Engine(1; J=NaN)
    @test_throws ArgumentError stationary_noise(e, 0.01, 1.0)
    @test_throws ArgumentError stationary_noise(e, Inf, 1)
    @test_throws ArgumentError simulate(e; T=1e-12, dt=0.01)
    @test_throws ArgumentError simulate(e; T=0.015, dt=0.01)
    @test_throws ArgumentError advance(
        e, initial_state(e), zeros(ComplexF64, 1), [NaN], [0.0], 0.01
    )
    @test size(stationary_noise(e, 0.01, 0)) == (1, 1)
end

@testset "Shared-noise nonlinear trajectory" begin
    e = Engine(
        2; J=0.4, delta=1.3, g=0.3, kappa=[0.1, 0.4], kmax=2, cutoff=1e-13, maxdim=64
    )
    psi = initial_state(e; preparation=:neel)
    shift = zeros(ComplexF64, 2)
    z = fixture("noise")
    discarded = Float64[]
    for i in 1:10
        psi, shift, disc = advance(e, psi, shift, z[2i - 1, :], z[2i, :], 0.01)
        append!(discarded, disc)
    end
    root, n2 = root_mps(e, psi)
    t = reduce(*, root)
    v = ComplexF64[t[e.sites[1] => a, e.sites[3] => b] for a in 1:2 for b in 1:2]
    @test abs(n2 - 1) < 1e-10
    @test maximum(abs.(v * v' - fixture("density"))) < 5e-5
    d = diagnostics(e, psi, discarded)
    @test !d.bond_cap_reached
    @test all(d.auxiliary_boundary_fractions .>= -1e-12)
    @test d.largest_discarded_weight < 1e-10
end

@testset "Single-mode support and covariance" begin
    e = Engine(1; kappa=0.4, kmax=2)
    r = simulate(e; T=0.02, dt=0.01, observe_every=1, preparation=:down)
    @test length(r.records) == 3
    @test all(isfinite, r.records[end].moments.pairs)
    z = stationary_noise(e, 0.01, 4, 2)
    @test size(z) == (1, 5)
    @test_throws ArgumentError Engine(2; kappa=[0.1, -0.2])
end
