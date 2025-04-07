
"""
    MPCPolicy(N, pomdp)

A policy that uses Model Predictive Control (MPC) to compute actions for a given POMDP.

# Fields
- `pomdp`: The POMDP model.
- `N::Int`: The planning horizon for MPC.
"""
@with_kw struct MPCPolicy <: POMDPs.Policy
    N::Int # = 10
    pomdp::iLQRPOMDP
end 

"""
    action_info(policy::MPCPolicy, b)

Compute the action using MPC for the given belief.

# Arguments
- `policy::MPCPolicy`: The MPC policy.
- `b`: The belief vector.

# Returns
- The optimal action computed by solving the MPC optimization problem.
"""
function action_info(policy::MPCPolicy, b)
    N = policy.N
    pomdp = policy.pomdp
    n_actions = num_actions(pomdp)
    num_states = num_states(pomdp)

    # Initial random actions for optimization
    initial_actions = [randn(n_actions) for _ in 1:N]
    flat_initial_actions = reduce(vcat, initial_actions)

    # Define the cost function
    function cost_function(flat_actions)
        actions = [flat_actions[(i-1)*n_actions+1:i*n_actions] for i in 1:N]
        belief = b
        total_cost = 0.0

        for t in 1:N
            action = actions[t]
            state = dyn_mean(pomdp, belief[1:num_states], action)  # Transition dynamics

            total_cost += cost(pomdp.Q, pomdp.R, pomdp.Q_N, state, action, pomdp.s_goal[1:num_states])
        end

        return total_cost
    end

    # Optimize using BFGS
    result = optimize(cost_function, flat_initial_actions, method=BFGS())
    opt_action = result.minimizer[1:n_actions]

    # Extract information about the optimization process
    total_cost = result.minimum
    predicted_actions = [result.minimizer[(i-1)*n_actions+1:i*n_actions] for i in 1:N]

    # Simulate the state trajectory for `action_info`
    predicted_states = []
    belief = initial_belief
    for action in predicted_actions
        state = dyn_mean(pomdp, belief[1:num_states(pomdp)], action)
        push!(predicted_states, state)
    end

    # Compile action_info
    action_info = Dict(
        :converged => result.converged,
        :iterations => result.iterations,
        :final_cost => total_cost,
        :predicted_states => predicted_states,
        :predicted_actions => predicted_actions
    )

    return opt_action, action_info
end

POMDPs.action(policy::MPCPolicy, b) = action_info(policy, b)[1]