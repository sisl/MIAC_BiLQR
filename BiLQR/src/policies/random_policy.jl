

"""
    struct RandomPolicy

A policy that generates random actions within specified ranges.

# Fields
- `action_ranges::Vector{Tuple{Float64, Float64}}`: A vector of tuples where each tuple specifies the range (min, max) for each action dimension.
"""
struct RandomPolicy <: POMDPs.Policy
    pomdp::iLQRPOMDP
    # action_ranges::Vector{Tuple{Float64, Float64}}
end

"""
    action_info(policy::RandomPolicy, b)

Generate a random action based on the specified ranges in the `RandomPolicy`.

# Arguments
- `policy::RandomPolicy`: The policy object specifying action ranges.
- `b`: The belief or state (not used in this random policy, but required by the POMDPs.jl interface).

# Returns
- A tuple `(action, action_info)` where:
  - `action`: The randomly generated action vector.
  - `action_info`: A dictionary containing additional information about the action generation.
"""
function action_info(policy::RandomPolicy, b)
    # Extract action ranges
    pomdp = policy.pomdp

    # Generate a random action for each dimension
    action = rand(num_actions(pomdp))

    # Compile action_info
    action_info = Dict(
        # :action_ranges => action_ranges,  # The ranges used to generate the actions
        # :num_actions => length(action_ranges),  # Number of action dimensions
        # :random_seed => Random.default_rng().state.seed,  # RNG seed for reproducibility
        # :distribution => "Uniform",  # Indicates the type of random distribution
        # :action => action  # The generated action (useful for logging)
    )

    return action, action_info
end

POMDPs.action(policy::RandomPolicy, b) = action_info(policy, b)[1]

# solver = QMDPSolver()
# policy = solve(solver, pomdp)
