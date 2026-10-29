module CorrelatedHOPS

using LinearAlgebra, Random, ITensors, ITensorMPS
export Engine,
    initial_state,
    root_mps,
    means,
    generator,
    advance,
    stationary_noise,
    diagnostics,
    spin_correlations,
    simulate

# Each physical spin and each hierarchy index has its own tensor. The physical
# basis is |down>, |up>. The hierarchy local dimension is kmax + 1.
ITensors.space(::SiteType"CHSpin") = 2
ITensors.space(::SiteType"CHAux"; kmax=4) = kmax + 1
function ITensors.op(::OpName{name}, ::SiteType"CHSpin", s::Index; kwargs...) where {name}
    opname = String(name)
    sm = ComplexF64[0 1; 0 0]
    sp = sm'
    a = if opname == "Id"
        Matrix{ComplexF64}(I, 2, 2)
    elseif opname == "Sm"
        sm
    elseif opname == "Sp"
        sp
    elseif opname == "Sx"
        (sm + sp) / 2
    elseif opname == "Sy"
        (sp - sm) / (2im)
    elseif opname == "Sz"
        diagm(ComplexF64[-0.5, 0.5])
    else
        throw(ArgumentError("Unknown spin operator $name"))
    end
    return ITensor(a, prime(s), dag(s))
end
function ITensors.op(::OpName{name}, ::SiteType"CHAux", s::Index; kwargs...) where {name}
    opname = String(name)
    d = dim(s)
    b = diagm(1 => sqrt.(1:(d - 1)))
    a = if opname == "Id"
        Matrix{ComplexF64}(I, d, d)
    elseif opname == "B"
        b
    elseif opname == "Bdag"
        b'
    elseif opname == "N"
        diagm(0:(d - 1))
    elseif opname == "Boundary"
        diagm([zeros(d - 1); 1])
    else
        throw(ArgumentError("Unknown auxiliary operator $name"))
    end
    return ITensor(a, prime(s), dag(s))
end

struct Engine
    sites::Vector{Index}
    W::Matrix{ComplexF64}
    g::Vector{Float64}
    counter_rotating::Float64
    coupling::Symbol
    omega_spin::Float64
    J::Float64
    delta::Float64
    kmax::Vector{Int}
    cutoff::Float64
    maxdim::Int
end

function integer_parameter(value, name; minimum=1)
    if !(value isa Integer) || value isa Bool || value < minimum
        throw(ArgumentError("$name must be an integer >= $minimum"))
    end
    return Int(value)
end

"""Local spin couplings and a passive hopping bath; kappa is NUMBER loss.

The spin Hamiltonian is omega_spin sum Sz + J sum(SxSx+SySy+delta SzSz).
coupling=:rabi gives F=g(Sm+counter_rotating Sp); :sx and :sz give g Sx/Sz.
Supply h to replace the nearest-neighbour bath Hamiltonian.
kmax may differ by mode. cutoff is the SVD discarded-weight tolerance;
maxdim is an additional bond cap. See docs/api.md for units and diagnostics.
"""
function Engine(
    n::Integer;
    omega_spin=0.1,
    omega_cavity=1.0,
    g=0.3,
    counter_rotating=0.25,
    J=0.0,
    delta=1.0,
    Jb=0.25,
    kappa=0.2,
    coupling=:rabi,
    h=nothing,
    kmax=4,
    cutoff=1e-8,
    maxdim=64,
)
    n = integer_parameter(n, "n")
    raw_bounds = kmax isa Number ? fill(kmax, n) : collect(kmax)
    length(raw_bounds) == n || throw(ArgumentError("one kmax per mode required"))
    ks = [integer_parameter(k, "kmax") for k in raw_bounds]
    maxdim = integer_parameter(maxdim, "maxdim")
    losses = kappa isa Number ? fill(Float64(kappa), n) : Float64.(kappa)
    gs = g isa Number ? fill(Float64(g), n) : Float64.(g)
    length(losses) == n && all(isfinite, losses) && all(losses .>= 0) ||
        throw(ArgumentError("invalid kappa"))
    length(gs) == n && all(isfinite, gs) || throw(ArgumentError("invalid g"))
    all(isfinite, (omega_spin, counter_rotating, J, delta)) ||
        throw(ArgumentError("spin parameters must be finite"))
    0 <= cutoff < 1 && maxdim >= 1 || throw(ArgumentError("invalid compression parameters"))
    coupling in (:rabi, :sx, :sz) || throw(ArgumentError("unknown coupling"))
    bath = h === nothing ? diagm(fill(ComplexF64(omega_cavity), n)) : ComplexF64.(h)
    if h === nothing
        for i in 1:(n - 1)
            bath[i, i + 1] = bath[i + 1, i] = Jb
        end
    end
    size(bath) == (n, n) && ishermitian(bath) && all(isfinite, bath) ||
        throw(ArgumentError("h must be Hermitian"))
    s = Index[]
    for i in 1:n
        push!(s, siteind("CHSpin"; addtags="n=$i"))
        push!(s, siteind("CHAux"; kmax=ks[i], addtags="n=$i"))
    end
    return Engine(
        s,
        diagm(losses ./ 2) + im * bath,
        gs,
        counter_rotating,
        coupling,
        omega_spin,
        J,
        delta,
        ks,
        cutoff,
        maxdim,
    )
end

"""Product spin state with empty hierarchy; :down, :neel or :x_neel."""
function initial_state(e::Engine; preparation=:down)
    preparation in (:down, :neel, :x_neel) || throw(ArgumentError("unknown preparation"))
    if preparation != :x_neel
        labels = ones(Int, length(e.sites))
        if preparation == :neel
            for i in eachindex(e.kmax)
                labels[2i - 1] = isodd(i) ? 2 : 1
            end
        end
        return MPS(e.sites, labels)
    end
    tensors = ITensor[]
    for i in eachindex(e.kmax)
        push!(tensors, ITensor(ComplexF64[1, isodd(i) ? 1 : -1] / sqrt(2), e.sites[2i - 1]))
        push!(tensors, onehot(e.sites[2i] => 1))
    end
    return MPS(tensors)
end

"""Return (root MPS, squared root norm), projecting auxiliary indices to zero."""
function root_mps(e::Engine, psi::MPS; normalize=true)
    r = MPS([
        psi[2i - 1] * psi[2i] * onehot(dag(e.sites[2i]) => 1) for i in eachindex(e.kmax)
    ])
    norm2 = real(inner(r, r))
    isfinite(norm2) && norm2 > 1e-100 || error("Invalid physical root norm")
    return normalize ? (r / sqrt(norm2), norm2) : (r, norm2)
end

"""Normalised root expectations beta_i = <F_i^dagger>, one per bath mode."""
function means(e::Engine, psi::MPS)
    r, _ = root_mps(e, psi)
    if e.coupling == :rabi
        return e.g .* (expect(r, "Sp") + e.counter_rotating .* expect(r, "Sm"))
    end
    return e.g .* expect(r, e.coupling == :sx ? "Sx" : "Sz")
end

function terms(e::Engine; adjoint=false)
    if e.coupling == :rabi
        if adjoint
            return [("Sp", 1.0), ("Sm", e.counter_rotating)]
        end
        return [("Sm", 1.0), ("Sp", e.counter_rotating)]
    end
    return [(e.coupling == :sx ? "Sx" : "Sz", 1.0)]
end

"""Build identity * I + scale * G as an MPO; coefficient vectors have n entries."""
function generator(e::Engine, shift, z, beta; identity=0.0, scale=1.0)
    os = OpSum()
    n = length(e.kmax)
    for i in 1:n
        spin = 2i - 1
        aux = 2i
        os += -im * scale * e.omega_spin, "Sz", spin
        os += -scale * e.W[i, i], "N", aux
        for (name, c) in terms(e)
            os += scale * e.g[i] * c * (conj(z[i]) + shift[i]), name, spin
            os += scale * e.g[i] * c, name, spin, "Bdag", aux
        end
        for (name, c) in terms(e; adjoint=true)
            os += -scale * e.g[i] * c, name, spin, "B", aux
        end
        os += scale * beta[i], "B", aux
        for j in 1:n
            i == j && continue
            iszero(e.W[i, j]) ||
                ITensorMPS.add!(os, (-scale * e.W[i, j], "Bdag", aux, "B", 2j))
        end
    end
    for i in 1:(n - 1)
        os += -im * scale * e.J / 2, "Sp", 2i - 1, "Sm", 2i + 1
        os += -im * scale * e.J / 2, "Sm", 2i - 1, "Sp", 2i + 1
        os += -im * scale * e.J * e.delta, "Sz", 2i - 1, "Sz", 2i + 1
    end
    iszero(identity) || ITensorMPS.add!(os, (identity, "Id", 1))
    return MPO(os, e.sites)
end

const TAYLOR4_ROOTS = eigvals(ComplexF64[0 0 0 -24; 1 0 0 -24; 0 1 0 -12; 0 0 1 -4])

function shift_propagator(W, dt)
    n = size(W, 1)
    a = zeros(ComplexF64, 2n, 2n)
    a[1:n, 1:n] = -conj.(W)
    a[1:n, (n + 1):(2n)] = Matrix{ComplexF64}(I, n, n)
    p = exp(dt * a)
    return p[1:n, 1:n], p[1:n, (n + 1):(2n)]
end

"""Four factorised polynomial MPOs, with one midpoint predictor MPO.

The polynomial approximates an exponential to degree four for frozen
coefficients. Nonlinear/stochastic coefficient freezing has a separate
timestep error. Compression uses both cutoff and maxdim.
Return (new state, new shift, discarded weights). Weights include the
predictor and all four factors. z0 and zm are noise vectors at the start
and midpoint; the input state, shift and noise are read without modification.
"""
function advance(e::Engine, psi::MPS, shift, z0, zm, dt)
    out, new_shift, discarded, _ = advance_step(e, psi, shift, z0, zm, dt)
    return out, new_shift, discarded
end

# Keep transient bond diagnostics internally while preserving advance's API.
function advance_step(e::Engine, psi::MPS, shift, z0, zm, dt)
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be positive"))
    n = length(e.kmax)
    for coefficients in (shift, z0, zm)
        size(coefficients) == (n,) && all(isfinite, coefficients) ||
            throw(ArgumentError("shift and noise must be finite vectors of length $n"))
    end
    discarded = Float64[]
    maximum_bond_dimension = maxlinkdim(psi)
    callback(; link, truncation_error) = push!(discarded, truncation_error)
    function compress(mpo, state)
        out = apply(mpo, state; cutoff=e.cutoff, maxdim=e.maxdim, alg="naive", callback)
        maximum_bond_dimension = max(maximum_bond_dimension, maxlinkdim(out))
        return out
    end
    beta0 = means(e, psi)
    half = shift_propagator(e.W, dt / 2)
    pred = compress(generator(e, shift, z0, beta0; identity=1.0, scale=dt / 2), psi)
    middle = half[1] * shift + half[2] * beta0
    beta = means(e, pred)
    out = psi
    for r in TAYLOR4_ROOTS
        factor = generator(e, middle, zm, beta; identity=1.0, scale=(-dt / r))
        out = compress(factor, out)
    end
    _, n2 = root_mps(e, out)
    out /= sqrt(n2)
    prop = shift_propagator(e.W, dt)
    new_shift = prop[1] * shift + prop[2] * beta
    all(isfinite, new_shift) || error("Nonfinite Girsanov shift")
    return out, new_shift, discarded, maximum_bond_dimension
end

"""Exact stationary OU samples with shape (modes, steps + 1); dt is the grid spacing."""
function stationary_noise(e::Engine, dt, steps, seed=0)
    steps = integer_parameter(steps, "steps"; minimum=0)
    isfinite(dt) && dt > 0 || throw(ArgumentError("invalid noise grid"))
    n = length(e.kmax)
    p = exp(-dt * e.W)
    q = Hermitian(Matrix{ComplexF64}(I, n, n) - p * p')
    eig = eigen(q)
    minimum(eig.values) >= -1e-11 || error("Invalid noise covariance")
    factor = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0)))
    rng = MersenneTwister(seed)
    draw() = (randn(rng, n) + im * randn(rng, n)) / sqrt(2)
    z = zeros(ComplexF64, n, steps + 1)
    z[:, 1] = draw()
    for i in 1:steps
        z[:, i + 1] = p * z[:, i] + factor * draw()
    end
    return z
end

"""Current bond/boundary snapshot and maximum of the supplied discarded weights."""
function diagnostics(e::Engine, psi::MPS, discarded=Float64[])
    boundary = real.(expect(psi, "Boundary"; sites=2:2:length(e.sites)))
    return (;
        bond_dimensions=linkdims(psi),
        bond_cap_reached=maxlinkdim(psi) >= e.maxdim,
        largest_discarded_weight=isempty(discarded) ? 0.0 : maximum(discarded),
        auxiliary_boundary_fractions=boundary,
    )
end

"""Return raw root moments. Average moments before forming connected correlators."""
function spin_correlations(e::Engine, psi::MPS; first="Sz", second="Sz")
    r, _ = root_mps(e, psi)
    pairs = if length(r) == 1
        reshape(expect(r, first * " * " * second), 1, 1)
    else
        correlation_matrix(r, first, second)
    end
    return (; singles_first=expect(r, first), singles_second=expect(r, second), pairs)
end

"""One trajectory, returning (state, shift, records).

Initial and final observations are always recorded. callback receives
(time, root moments, diagnostics). Compression extrema and bond_cap_reached
are cumulative from the initial state, including predictor/factor applications
and steps between observations. Bond dimensions and boundary fractions are
current-state snapshots. See docs/api.md for moment array conventions.
"""
function simulate(
    e::Engine; T=1.0, dt=0.01, seed=0, preparation=:neel, observe_every=10, callback=nothing
)
    observe_every = integer_parameter(observe_every, "observe_every")
    isfinite(T) && isfinite(dt) && T > 0 && dt > 0 ||
        throw(ArgumentError("invalid time grid"))
    grid_steps = T / dt
    isfinite(grid_steps) && grid_steps < typemax(Int) ÷ 2 ||
        throw(ArgumentError("too many time steps"))
    steps = round(Int, grid_steps)
    steps >= 1 && isapprox(grid_steps, steps; atol=1e-9, rtol=0) ||
        throw(ArgumentError("T must lie on the internal grid"))
    z = stationary_noise(e, dt / 2, 2steps, seed)
    psi = initial_state(e; preparation)
    shift = zeros(ComplexF64, length(e.kmax))
    records = []
    maximum_bond_dimension = maxlinkdim(psi)
    largest_discarded_weight = 0.0
    function record(t)
        snapshot = diagnostics(e, psi)
        cumulative = (;
            snapshot...,
            maximum_bond_dimension,
            bond_cap_reached=maximum_bond_dimension >= e.maxdim,
            largest_discarded_weight,
        )
        item = (; time=t, moments=spin_correlations(e, psi), diagnostics=cumulative)
        push!(records, item)
        if callback !== nothing
            callback(t, item.moments, item.diagnostics)
        end
    end
    record(0.0)
    for i in 1:steps
        psi, shift, disc, step_bond = advance_step(
            e, psi, shift, z[:, 2i - 1], z[:, 2i], dt
        )
        maximum_bond_dimension = max(maximum_bond_dimension, step_bond)
        if !isempty(disc)
            largest_discarded_weight = max(largest_discarded_weight, maximum(disc))
        end
        if i % observe_every == 0 || i == steps
            record(i * dt)
        end
    end
    return (; state=psi, shift, records)
end
end
