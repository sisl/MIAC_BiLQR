# using LinearAlgebra
using BiLQR
using Random 

rng = MersenneTwister(1) 

println("Testing...")

pomdp = create_sample_cartpole()

horizon = 100
N = 10
eps = 1e-6
max_iters = 100

policy = BiLQRPolicy(pomdp, N, eps, max_iters)

belief_updater = EKFUpdater(pomdp)

all_b, info_dict = simulate(horizon, policy, belief_updater)

# # test observation function 
# # Define a state and action
# state = [0.0, π / 4, 0.0, 0.0, 2.0]
# action = [0.1]

# # Call the POMDPs.transition function
# transition_distribution = POMDPs.transition(pomdp, state, action)

# # Print the result
# println("Transition distribution mean: ", mean(transition_distribution))
# # println("Transition distribution covariance: ", cov(transition_distribution))

# # test observation function
# observation_distribution = POMDPs.observation(pomdp, state, action)

# # Print the result
# println("Observation distribution mean: ", mean(observation_distribution))
# # println("Observation distribution covariance: ", cov(observation_distribution))







# all_b, info_dict = simulate(horizon, policy, belief_updater)

# Test valid input
# @testset "CartpoleMDP Constructor Tests" begin
#     num_states = 5
#     num_actions = 1
#     num_observations = 4

#     Q = diagm(ones(num_states))
#     Q_N = diagm(ones(num_states))
#     R = diagm(ones(num_actions))
#     Λ = diagm(ones(num_states))
#     m0 = zeros(num_states)
#     Σ0 = diagm(ones(num_states))
#     δt = 0.1
#     mc = 1.0
#     g = 9.81
#     l = 1.0
#     W_state_process = diagm(ones(num_states))
#     W_process = diagm(ones(num_states))
#     W_obs = diagm(ones(num_observations))
#     W_obs_ekf = diagm(ones(num_observations))

#     # Valid instantiation
#     @test CartpoleMDP(Q, R, Q_N, Λ, m0, Σ0, δt, mc, g, l, W_state_process, W_process, W_obs, W_obs_ekf) isa CartpoleMDP

#     # Invalid Q dimensions
#     Q_invalid = diagm(ones(num_states - 1))
#     @test_throws AssertionError CartpoleMDP(Q_invalid, R, Q_N, Λ, m0, Σ0, δt, mc, g, l, W_state_process, W_process, W_obs, W_obs_ekf)

#     # Invalid R dimensions
#     R_invalid = diagm(ones(num_actions + 1))
#     @test_throws AssertionError CartpoleMDP(Q, R_invalid, Q_N, Λ, m0, Σ0, δt, mc, g, l, W_state_process, W_process, W_obs, W_obs_ekf)

#     # Invalid m0 length
#     m0_invalid = zeros(num_states + 1)
#     @test_throws AssertionError CartpoleMDP(Q, R, Q_N, Λ, m0_invalid, Σ0, δt, mc, g, l, W_state_process, W_process, W_obs, W_obs_ekf)
# end


