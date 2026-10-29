using Test, CorrelatedHOPS, ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles
BLAS.set_num_threads(1)
fixture(name)=readdlm(joinpath(@__DIR__,"fixtures",name*"_real.csv"),',',Float64)+
              im*readdlm(joinpath(@__DIR__,"fixtures",name*"_imag.csv"),',',Float64)

@testset "Local representation and Python generator" begin
    e=Engine(2;J=.4,delta=1.3,g=.3,kappa=[.1,.4],kmax=[1,2],cutoff=1e-13,maxdim=64)
    @test dim.(e.sites)==[2,2,2,3]
    shift=ComplexF64[.01+.02im,-.02+.03im];z=ComplexF64[.2+.1im,-.3+.2im]
    beta=ComplexF64[.02+.01im,.01-.02im]
    mpo=generator(e,shift,z,beta); t=reduce(*,mpo)
    # Python C-order basis: spin1, aux1, spin2, aux2 (last index fastest).
    labels=[(s1,k1,s2,k2) for s1 in 1:2 for k1 in 1:2 for s2 in 1:2 for k2 in 1:3]
    mat=zeros(ComplexF64,length(labels),length(labels))
    for (r,rr) in enumerate(labels),(c,cc) in enumerate(labels)
        inds=[prime(e.sites[i])=>rr[i] for i in 1:4]
        append!(inds,[dag(e.sites[i])=>cc[i] for i in 1:4]);mat[r,c]=t[inds...]
    end
    @test maximum(abs.(mat-fixture("generator"))) < 1e-12
    psi=initial_state(e;preparation=:neel);root,n2=root_mps(e,psi)
    @test n2 ≈ 1
    @test expect(root,"Sz") ≈ [.5,-.5]
end

@testset "Shared-noise nonlinear trajectory" begin
    e=Engine(2;J=.4,delta=1.3,g=.3,kappa=[.1,.4],kmax=2,cutoff=1e-13,maxdim=64)
    psi=initial_state(e;preparation=:neel);shift=zeros(ComplexF64,2);z=fixture("noise")
    discarded=Float64[]
    for i in 1:10
        psi,shift,disc=advance(e,psi,shift,z[2i-1,:],z[2i,:],.01)
        append!(discarded,disc)
    end
    root,n2=root_mps(e,psi);t=reduce(*,root)
    v=ComplexF64[t[e.sites[1]=>a,e.sites[3]=>b] for a in 1:2 for b in 1:2]
    @test abs(n2-1)<1e-10
    @test maximum(abs.(v*v'-fixture("density"))) < 5e-5
    d=diagnostics(e,psi,discarded)
    @test !d.bond_cap_reached
    @test all(d.auxiliary_boundary_fractions .>= -1e-12)
    @test d.largest_discarded_weight < 1e-10
end

@testset "Single-mode support and covariance" begin
    e=Engine(1;kappa=.4,kmax=2)
    r=simulate(e;T=.02,dt=.01,observe_every=1,preparation=:down)
    @test length(r.records)==3
    @test all(isfinite,r.records[end].moments.pairs)
    z=stationary_noise(e,.01,4,2)
    @test size(z)==(1,5)
    @test_throws ArgumentError Engine(2;kappa=[.1,-.2])
end
