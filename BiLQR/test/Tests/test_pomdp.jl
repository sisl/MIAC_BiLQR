
function create_cartpole()
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
    return cartpole_mdp
end 

# Check the initialized object
println(cartpole_mdp)

# test transition function 

# Define a state and action
state = [0.0, π / 4, 0.0, 0.0, 2.0]
action = [0.1]

# Call the POMDPs.transition function
transition_distribution = POMDPs.transition(cartpole_mdp, state, action)

# Print the result
println("Transition distribution mean: ", mean(transition_distribution))
println("Transition distribution covariance: ", cov(transition_distribution))

# test observation function
observation_distribution = POMDPs.observation(cartpole_mdp, state, action)

# Print the result
println("Observation distribution mean: ", mean(observation_distribution))
println("Observation distribution covariance: ", cov(observation_distribution))