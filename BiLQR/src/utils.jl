using POMDPs
using POMDPTools
using JLD2 

"""
    create_sample_cartpole()

Create a sample `CartpoleMDP` instance with predefined parameters.

# Returns
- `CartpoleMDP`: An instance of the `CartpoleMDP` with predefined matrices and parameters.

# Example
```julia
pomdp = create_sample_cartpole()
```
"""
function create_sample_cartpole()
    # Define the matrices and parameters
    Q = 1e-4 * I(5)
    R = 1e-4 * I(1)
    Q_N = Diagonal([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    Λ = Diagonal(vcat(fill(1e-4, 24), [1]))  # 5^2

    m0 = [0.0, π / 2, 0.0, 0.0, 2.0]  # Initial mean of the belief
    Σ0 = [1e-4, 1e-4, 1e-4, 1e-4, 2.0]  # Initial covariance matrix

    δt = 0.1        # Time step
    mc = 1.0        # Cart mass
    g = 9.81        # Gravitational acceleration
    l = 1.0         # Pole length

    # Noise covariance matrices
    W_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    W_obs = diagm([1e-2, 1e-2, 1e-2, 1e-2])

    # Number of states, actions, observations, and system variables
    num_states = 5
    num_actions = 1
    num_observations = 4
    num_sysvars = 1

    cartpole_mdp = CartpoleMDP(Q, R, Q_N, Λ, m0, Σ0, δt, mc, g, l, W_process, W_obs, 
                                num_states, num_actions, num_observations, num_sysvars)
    return cartpole_mdp
end 

"""
    simulate(time_steps::Int, policy, belief_updater)

Simulates system identification for the Cartpole using the given policy.

# Arguments
- `time_steps::Int`: Number of simulation steps.
- `policy`: The policy to be used (e.g., BiLQR, MPC, etc.).
- `belief_updater`: The belief updater used to refine the state belief.

# Returns
- A tuple `(all_b, info_dict)`, where:
  - `all_b`: List of belief states at each time step.
  - `info_dict`: Dictionary containing estimated means, variances, control inputs, and true parameters.
"""
function simulate(time_steps::Int, policy, belief_updater)
    pomdp = policy.pomdp

    # keep the \Sigmas as part of the MDP itself and have the actual state as the thing that gets sampled from the belief 
    s = pomdp.s_init
    b = vcat(s[1:end - pomdp.num_sysvars], pomdp.s_init[pomdp.num_states - pomdp.num_sysvars + 1:end], diagm(pomdp.Σ0)[:])

    # Data storage
    vec_estimates = [b[pomdp.num_states - pomdp.num_sysvars + 1:pomdp.num_states]]
    variances = [diagm(b[end - pomdp.num_sysvars + 1:end])]
    all_s = [s]
    all_b = [b]
    all_u = []
    means = []
    variances = []

    for t in 1:time_steps
        println("timestep: ", t)
        
        push!(all_b, b)
        push!(all_s, s)

        a, action_dict = action_info(policy, b)
        push!(all_u, a)

        s = rand(POMDPs.transition(pomdp, s, a))
        z = rand(POMDPs.observation(pomdp, s, a))

        b = update(belief_updater, b, a, z)
        
        # if b === nothing
        #     println("Belief update failed; terminating simulation.")
        #     return nothing
        # end

        push!(means, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)])
        push!(variances, diagm(b[end - num_sysvars(pomdp) + 1:end]))
    end

    ΣΘΘ = b[end]
    
    info_dict = Dict(:all_b => all_b, :means => means, :variances => variances, :ΣΘΘ => ΣΘΘ, :all_s => all_s, :all_u => all_u, :true_params => pomdp.mp_true)
    return all_b, info_dict
end

"""
    save_data(jld2_file::String, info_dict::Dict)

Save simulation data to a JLD2 file.

# Arguments
- `jld2_file::String`: Filepath to save the data.
- `info_dict::Dict`: Dictionary containing simulation results.

# Example
```julia
save_data("simulation_data.jld2", info_dict)
```
"""
function save_data(jld2_file::String, info_dict::Dict)
    @save jld2_file info_dict
end 

"""
    load_data(jld2_file::String)

Load simulation data from a JLD2 file.

# Arguments
- `jld2_file::String`: Filepath from which to load the data.

# Returns
- `info_dict::Dict`: Dictionary containing the loaded simulation results.

# Example
```julia
info_dict = load_data("simulation_data.jld2")
```
"""
function load_data(jld2_file::String)
    info_dict = Dict()
    @load jld2_file info_dict
    return info_dict
end

function compute_mse(estimated_params, true_params)
    return sum((estimated_params - true_params).^2) / length(true_params)
end

function plotting()
end 
