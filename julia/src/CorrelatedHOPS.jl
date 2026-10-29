module CorrelatedHOPS

using LinearAlgebra, Random, ITensors, ITensorMPS
export Engine, initial_state, root_mps, means, generator, advance, stationary_noise,
       diagnostics, spin_correlations, simulate

# Each physical spin and each hierarchy index has its own tensor. The physical
# basis is |down>, |up>. The hierarchy local dimension is kmax + 1.
ITensors.space(::SiteType"CHSpin") = 2
ITensors.space(::SiteType"CHAux"; kmax=4) = kmax + 1
function ITensors.op(::OpName{name}, ::SiteType"CHSpin", s::Index; kwargs...) where {name}
    opname = String(name)
    sm = ComplexF64[0 1; 0 0]; sp = sm'
    a = opname == "Id" ? Matrix{ComplexF64}(I,2,2) :
        opname == "Sm" ? sm : opname == "Sp" ? sp :
        opname == "Sx" ? (sm+sp)/2 : opname == "Sy" ? (sp-sm)/(2im) :
        opname == "Sz" ? diagm(ComplexF64[-.5,.5]) : error("Unknown spin operator $name")
    ITensor(a, prime(s), dag(s))
end
function ITensors.op(::OpName{name}, ::SiteType"CHAux", s::Index; kwargs...) where {name}
    opname = String(name)
    d=dim(s); b=diagm(1 => sqrt.(1:d-1))
    a = opname == "Id" ? Matrix{ComplexF64}(I,d,d) : opname == "B" ? b :
        opname == "Bdag" ? b' : opname == "N" ? diagm(0:d-1) :
        opname == "Boundary" ? diagm([zeros(d-1);1]) : error("Unknown auxiliary operator $name")
    ITensor(a, prime(s), dag(s))
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

"""Local spin couplings and a passive hopping bath; kappa is NUMBER loss.

The spin Hamiltonian is omega_spin sum Sz + J sum(SxSx+SySy+delta SzSz).
coupling=:rabi gives F=g(Sm+counter_rotating Sp); :sx and :sz give g Sx/Sz.
Supply h to replace the nearest-neighbour bath Hamiltonian.
"""
function Engine(n::Integer; omega_spin=.1, omega_cavity=1., g=.3,
                counter_rotating=.25, J=0., delta=1., Jb=.25, kappa=.2,
                coupling=:rabi, h=nothing, kmax=4, cutoff=1e-8, maxdim=64)
    n>=1 || throw(ArgumentError("n must be positive"))
    ks=kmax isa Integer ? fill(Int(kmax),n) : Int.(kmax)
    length(ks)==n && all(ks .>= 1) || throw(ArgumentError("one positive kmax per mode required"))
    losses=kappa isa Number ? fill(Float64(kappa),n) : Float64.(kappa)
    gs=g isa Number ? fill(Float64(g),n) : Float64.(g)
    length(losses)==n && all(isfinite,losses) && all(losses .>= 0) || throw(ArgumentError("invalid kappa"))
    length(gs)==n && all(isfinite,gs) || throw(ArgumentError("invalid g"))
    0<=cutoff<1 && maxdim>=1 || throw(ArgumentError("invalid compression parameters"))
    coupling in (:rabi,:sx,:sz) || throw(ArgumentError("unknown coupling"))
    bath=h===nothing ? diagm(fill(ComplexF64(omega_cavity),n)) : ComplexF64.(h)
    if h===nothing
        for i in 1:n-1; bath[i,i+1]=bath[i+1,i]=Jb; end
    end
    size(bath)==(n,n) && ishermitian(bath) && all(isfinite,bath) || throw(ArgumentError("h must be Hermitian"))
    s=Index[]
    for i in 1:n
        push!(s,siteind("CHSpin";addtags="n=$i"))
        push!(s,siteind("CHAux";kmax=ks[i],addtags="n=$i"))
    end
    Engine(s,diagm(losses./2)+im*bath,gs,counter_rotating,coupling,
           omega_spin,J,delta,ks,cutoff,maxdim)
end

"""Product spin state with empty hierarchy; :down, :neel or :x_neel."""
function initial_state(e::Engine; preparation=:down)
    preparation in (:down,:neel,:x_neel) || throw(ArgumentError("unknown preparation"))
    if preparation != :x_neel
        labels=ones(Int,length(e.sites))
        if preparation==:neel
            for i in eachindex(e.kmax); labels[2i-1]=isodd(i) ? 2 : 1; end
        end
        return MPS(e.sites,labels)
    end
    tensors=ITensor[]
    for i in eachindex(e.kmax)
        push!(tensors,ITensor(ComplexF64[1,isodd(i) ? 1 : -1]/sqrt(2),e.sites[2i-1]))
        push!(tensors,onehot(e.sites[2i]=>1))
    end
    MPS(tensors)
end

"""Project all auxiliary indices to zero. They are not physical bath states."""
function root_mps(e::Engine,psi::MPS; normalize=true)
    r=MPS([psi[2i-1]*psi[2i]*onehot(dag(e.sites[2i])=>1) for i in eachindex(e.kmax)])
    norm2=real(inner(r,r))
    isfinite(norm2) && norm2>1e-100 || error("Invalid physical root norm")
    normalize ? (r/sqrt(norm2),norm2) : (r,norm2)
end

function means(e::Engine,psi::MPS)
    r,_=root_mps(e,psi)
    e.coupling==:rabi ? e.g.*(expect(r,"Sp")+e.counter_rotating.*expect(r,"Sm")) :
        e.g.*expect(r,e.coupling==:sx ? "Sx" : "Sz")
end

function terms(e::Engine; adjoint=false)
    e.coupling==:rabi ? (adjoint ? [("Sp",1.),("Sm",e.counter_rotating)] :
        [("Sm",1.),("Sp",e.counter_rotating)]) : [(e.coupling==:sx ? "Sx" : "Sz",1.)]
end

function generator(e::Engine,shift,z,beta; identity=0., scale=1.)
    os=OpSum();n=length(e.kmax)
    for i in 1:n
        spin=2i-1; aux=2i
        os+=-im*scale*e.omega_spin,"Sz",spin
        os+=-scale*e.W[i,i],"N",aux
        for (name,c) in terms(e)
            os+=scale*e.g[i]*c*(conj(z[i])+shift[i]),name,spin
            os+=scale*e.g[i]*c,name,spin,"Bdag",aux
        end
        for (name,c) in terms(e;adjoint=true)
            os+=-scale*e.g[i]*c,name,spin,"B",aux
        end
        os+=scale*beta[i],"B",aux
        for j in 1:n
            i==j && continue
            iszero(e.W[i,j]) || ITensorMPS.add!(os,(-scale*e.W[i,j],"Bdag",aux,"B",2j))
        end
    end
    for i in 1:n-1
        os+=-im*scale*e.J/2,"Sp",2i-1,"Sm",2i+1
        os+=-im*scale*e.J/2,"Sm",2i-1,"Sp",2i+1
        os+=-im*scale*e.J*e.delta,"Sz",2i-1,"Sz",2i+1
    end
    iszero(identity) || ITensorMPS.add!(os,(identity,"Id",1))
    MPO(os,e.sites)
end

const TAYLOR4_ROOTS=eigvals(ComplexF64[0 0 0 -24;1 0 0 -24;0 1 0 -12;0 0 1 -4])

function shift_propagator(W,dt)
    n=size(W,1); a=zeros(ComplexF64,2n,2n)
    a[1:n,1:n]=-conj.(W);a[1:n,n+1:2n]=Matrix{ComplexF64}(I,n,n)
    p=exp(dt*a); (p[1:n,1:n],p[1:n,n+1:2n])
end

"""Four factorised polynomial MPOs, with one midpoint predictor MPO.

The polynomial approximates an exponential to degree four for frozen
coefficients. Nonlinear/stochastic coefficient freezing has a separate
timestep error. Compression uses both cutoff and maxdim.
"""
function advance(e::Engine,psi::MPS,shift,z0,zm,dt)
    isfinite(dt) && dt>0 || throw(ArgumentError("dt must be positive"))
    beta0=means(e,psi); half=shift_propagator(e.W,dt/2)
    pred=apply(generator(e,shift,z0,beta0;identity=1.,scale=dt/2),psi;
               cutoff=e.cutoff,maxdim=e.maxdim,alg="naive")
    middle=half[1]*shift+half[2]*beta0; beta=means(e,pred)
    out=psi;discarded=Float64[]
    callback(;link,truncation_error)=push!(discarded,truncation_error)
    for r in TAYLOR4_ROOTS
        factor=generator(e,middle,zm,beta;identity=1.,scale=-dt/r)
        out=apply(factor,out;cutoff=e.cutoff,maxdim=e.maxdim,alg="naive",callback)
    end
    _,n2=root_mps(e,out);out/=sqrt(n2)
    prop=shift_propagator(e.W,dt)
    out,prop[1]*shift+prop[2]*beta,discarded
end

"""Exact stationary OU samples, columns are times; supports unequal losses."""
function stationary_noise(e::Engine,dt,steps,seed=0)
    dt>0 && steps>=0 || throw(ArgumentError("invalid noise grid"))
    n=length(e.kmax);p=exp(-dt*e.W);q=Hermitian(Matrix{ComplexF64}(I,n,n)-p*p')
    eig=eigen(q);minimum(eig.values)>=-1e-11 || error("Invalid noise covariance")
    factor=eig.vectors*Diagonal(sqrt.(max.(eig.values,0)))
    rng=MersenneTwister(seed);draw()=(randn(rng,n)+im*randn(rng,n))/sqrt(2)
    z=zeros(ComplexF64,n,steps+1);z[:,1]=draw()
    for i in 1:steps;z[:,i+1]=p*z[:,i]+factor*draw();end
    z
end

function diagnostics(e::Engine,psi::MPS,discarded=Float64[])
    boundary=real.(expect(psi,"Boundary";sites=2:2:length(e.sites)))
    (;bond_dimensions=linkdims(psi),bond_cap_reached=maxlinkdim(psi)>=e.maxdim,
      largest_discarded_weight=isempty(discarded) ? 0. : maximum(discarded),
      auxiliary_boundary_fractions=boundary)
end

"""Return raw root moments. Average moments before forming connected correlators."""
function spin_correlations(e::Engine,psi::MPS; first="Sz",second="Sz")
    r,_=root_mps(e,psi)
    pairs=length(r)==1 ? reshape(expect(r,first*" * "*second),1,1) :
        correlation_matrix(r,first,second)
    (;singles_first=expect(r,first),singles_second=expect(r,second),
      pairs)
end

"""One trajectory; callback receives (time, root moments, diagnostics)."""
function simulate(e::Engine;T=1.,dt=.01,seed=0,preparation=:neel,observe_every=10,callback=nothing)
    steps=round(Int,T/dt)
    T>0 && dt>0 && isapprox(steps*dt,T) && observe_every>=1 || throw(ArgumentError("invalid time grid"))
    z=stationary_noise(e,dt/2,2steps,seed);psi=initial_state(e;preparation)
    shift=zeros(ComplexF64,length(e.kmax));records=[]
    function record(t,disc)
        item=(;time=t,moments=spin_correlations(e,psi),diagnostics=diagnostics(e,psi,disc))
        push!(records,item);callback===nothing || callback(t,item.moments,item.diagnostics)
    end
    record(0.,Float64[])
    for i in 1:steps
        psi,shift,disc=advance(e,psi,shift,z[:,2i-1],z[:,2i],dt)
        (i%observe_every==0 || i==steps) && record(i*dt,disc)
    end
    (;state=psi,shift,records)
end
end
