module BiLQR

# greet() = print("Hello World!")
using POMDPs
using POMDPModels
using POMDPTools

using LinearAlgebra
using ForwardDiff
using Distributions
using SparseArrays
using Optim
using Parameters 
using Random
using Revise 

export
    iLQRPOMDP,
    dyn_mean,
    dyn_noise,
    obs_mean,
    obs_noise,
    num_states,
    num_actions,
    num_observations,
    num_sysvars

include("ilqr_pomdp.jl")

export 
    CartpoleMDP,
    AirplaneMDP 

include("./problems/cartpole_pomdp.jl")
include("./problems/airplane_pomdp.jl")

export
    BiLQRPolicy,
    bilqr, 
    MPCPolicy,
    RandomPolicy

include("./policies/bilqr_policy.jl")
include("./policies/mpc_policy.jl")
include("./policies/random_policy.jl")

export
    update,
    EKFUpdater

include("./updaters/ekf.jl")
# include("./updaters/regression.jl")

export 
    create_sample_cartpole,
    simulate
    # plotting 
    # saving 
    # loading 

include("./utils.jl")

end # module BiLQR
