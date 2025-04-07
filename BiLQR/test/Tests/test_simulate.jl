using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../POMDPs/cartpole_ilqrpomdp.jl")
include("../Policies/bilqr_policy.jl")

# Define the matrices and parameters
Q = diagm([1.0, 1.0, 1.0, 1.0, 1.0])
R = diagm([1.0])
Q_N = diagm([1.0, 1.0, 1.0, 1.0, 1.0])
Λ = diagm([1.0, 1.0, 1.0, 1.0, 1.0])

m0 = [0.0, π / 2, 0.0, 0.0, 2.0]         # Initial mean of the belief
Σ0 = diagm([1e-4, 1e-4, 1e-4, 1e-4, 2.0])  # Initial covariance matrix

δt = 0.1        # Time step
mc = 1.0        # Cart mass
g = 9.81        # Gravitational acceleration
l = 1.0         # Pole length

# Noise covariance matrices
W_state_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
W_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
W_obs = diagm([1e-2, 1e-2, 1e-2, 1e-2])
W_obs_ekf = diagm([1e-2, 1e-2, 1e-2, 1e-2])

# Create the CartpoleMDP instance
cartpole_mdp = CartpoleMDP(
    Q, R, Q_N, Λ, m0, Σ0, δt, mc, g, l,
    W_state_process, W_process, W_obs, W_obs_ekf
)

horizon = 100
N = 10
eps = 1e-6
max_iters = 100

policy = BiLQRPolicy(pomdp = cartpole_mdp, N = N, eps = eps, max_iters = max_iters)
simulate(pomdp::CartpoleMDP, policy.max_iters, policy)




#     for (current_s, a, z, r) in stepthrough(pomdp, policy, belief_updater, b, s, "s,a,o,r", max_steps=time_steps)

#         println("Time step: $i")

#         # Store the action
#         push!(all_u, a)

#         # Update the belief using the updater
#         b = belief_updater.update(belief_updater, b, a, z)
#         if b === nothing
#             println("Belief update failed; terminating simulation.")
#             return nothing
#         end

#         # Extract mean and covariance from belief
#         m = b[1:num_states(pomdp)]
#         Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))

#         # Store estimates
#         push!(means, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)])
#         push!(variances, diagm(b[end - num_sysvars(pomdp) + 1:end]))
#         push!(all_b, b)

#         # Log the current state
#         push!(all_s, current_s)

#         i += 1
#     end

#     ΣΘΘ = b[end]
#     info_dict = Dict(:all_b => all_b, :means => means, :variances => variances, :ΣΘΘ => ΣΘΘ, :all_s => all_s, :all_u => all_u, :true_params => pomdp.true_params)
#     return all_b, info_dict

# """
#     validate_dimensions(Q, Q_N, R, Λ, m0, Σ0, W_state_process, W_process, W_obs, W_obs_ekf, num_states, num_actions, num_observations, num_sysvars)

# Validate the dimensions of the matrices and vectors for the CartpoleMDP.

# # Arguments
# - `Q`: The state cost matrix.
# - `Q_N`: The terminal state cost matrix.
# - `R`: The control cost matrix.
# - `Λ`: The state regularization matrix.
# - `m0`: The initial mean vector.
# - `Σ0`: The initial covariance matrix.
# - `W_state_process`: The state process noise covariance matrix.
# - `W_process`: The process noise covariance matrix.
# - `W_obs`: The observation noise covariance matrix.
# - `W_obs_ekf`: The observation noise covariance matrix for the EKF.
# - `num_states`: The number of states.
# - `num_actions`: The number of actions.
# - `num_observations`: The number of observations.
# - `num_sysvars`: The number of system variables.
# """

# function validate_dimensions(Q, Q_N, R, Λ, m0, Σ0, W_state_process, W_process, W_obs, W_obs_ekf, num_states, num_actions, num_observations, num_sysvars)
#     @assert size(Q) == (num_states, num_states) "Q matrix must be ${num_states}x${num_states}"
#     @assert size(Q_N) == (num_states, num_states) "Q_N matrix must be ${num_states}x${num_states}"
#     @assert size(R) == (num_actions, num_actions) "R matrix must be ${num_actions}x${num_actions}"
#     @assert size(Λ) == (num_states, num_states) "Λ matrix must be ${num_states}x${num_states}"
#     @assert length(m0) == num_states "Initial mean vector (m0) must have length ${num_states}"
#     @assert size(Σ0) == (num_states, num_states) "Initial covariance matrix (Σ0) must be ${num_states}x${num_states}"
#     @assert size(W_state_process) == (num_states, num_states) "W_state_process must be ${num_states}x${num_states}"
#     @assert size(W_process) == (num_states, num_states) "W_process must be ${num_states}x${num_states}"
#     @assert size(W_obs) == (num_observations, num_observations) "W_obs must be ${num_observations}x${num_observations}"
#     @assert size(W_obs_ekf) == (num_observations, num_observations) "W_obs_ekf must be ${num_observations}x${num_observations}"
# end
