

## POMDP type

abstract type iLQRPOMDP{S,A,O} <: POMDP{S,A,O} end

# interface
"""
    dyn_mean(p::iLQRPOMDP, s::AbstractVector, a::AbstractVector)::AbstractVector

    Return the mean dynamics update from state `s` with control `a`.
"""
function dyn_mean end

"""
    dyn_noise(p::iLQRPOMDP, s::AbstractVector, a::AbstractVector)::AbstractMatrix

    Return the covariance of the dynamics update from state `s` with control `a`.
"""
function dyn_noise end

"""
    obs_mean(p::iLQRPOMDP, sp::AbstractVector)::AbstractVector

    Return the mean observation from state `sp`.
"""
function obs_mean end

"""
    obs_noise(p::iLQRPOMDP, sp::AbstractVector)::AbstractMatrix

    Return the covariance of the observation from state `sp`.
"""
function obs_noise end 

"""
    num_states(p::iLQRPOMDP)::Int

    Return the dimensionality of the state space in the POMDP.
"""
num_truestates(p::iLQRPOMDP) = p.num_states # throw exceptions if not defined as an integer 

"""
    num_actions(p::iLQRPOMDP)::Int

    Return the dimensionality of the action space in the POMDP.
"""
num_actions(p::iLQRPOMDP) = p.num_actions

"""
    num_observations(p::iLQRPOMDP)::Int

    Return the dimensionality of the observation space in the POMDP.
"""
num_observations(p::iLQRPOMDP) = p.num_observations

"""
    num_sysvars(p::iLQRPOMDP)::Int

    Return the dimensionality of the system variables in the POMDP.
"""
num_sysvars(p::iLQRPOMDP) = p.num_sysvars

POMDPs.transition(p::iLQRPOMDP, s, a) = MvNormal(dyn_mean(p, s, a), dyn_noise(p, s, a))
POMDPs.observation(p::iLQRPOMDP, s, a) = MvNormal(obs_mean(p, s, a), obs_noise(p, s, a))

### consistent policy interface for later user
