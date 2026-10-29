using CorrelatedHOPS, LinearAlgebra
BLAS.set_num_threads(1)

# A short demonstration. Increase trajectory count and validate numerical
# parameters for a production calculation. kappa is the NUMBER decay rate.
engine=Engine(4;J=1.,delta=2.,coupling=:sx,g=.5,omega_spin=0.,
              omega_cavity=3.,Jb=.25,kappa=.4,kmax=3,cutoff=1e-8,maxdim=64)
result=simulate(engine;T=.1,dt=.01,seed=7,preparation=:neel,observe_every=5,
    callback=(t,m,d)->println((;t,maxbond=maximum(d.bond_dimensions),
                              cap_reached=d.bond_cap_reached)))
println("Average the returned raw moments over trajectories before forming connected correlations.")
