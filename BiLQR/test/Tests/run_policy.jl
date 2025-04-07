
"""
    simulate(pomdp::iLQRPOMDP, num_steps, policy)

Simulates system identification for the Cartpole using the given policy.

# Arguments
- `pomdp`: The Cartpole system identification POMDP.
- `num_steps`: Number of simulation steps.
- `policy`: The policy to be used (e.g., BiLQR, MPC, etc.).

# Returns
- A tuple `(all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, mp_true)`.
"""

##TODO: can use stepthrough to get the true state and action
# stepthrough(pomdp::POMDP, policy::Policy, [up::Updater, [initial_belief, [initial_state]]], [spec]; [kwargs...])
# pomdp = BabyPOMDP()
# policy = RandomPolicy(pomdp)

# for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps=10)
#     println("in state $s")
#     println("took action $o")
#     println("received observation $o and reward $r")
# end

function simulate(time_steps::Int, policy, belief_updater)
    rng = Random.default_rng()
    pomdp = policy.pomdp # TODO: ensure policy.pomdp = belief_updater.pomdp? 

    b = initialstate_distribution(pomdp).support[1]
    s = pommdp.s_init

    # Data storage
    vec_estimates = [b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)]]
    variances = [diagm(b[end-num_sysvars(pomdp) + 1:end])]
    all_s = [s]
    all_b = [b]
    all_u = []
    means = []
    variances = []

    # Simulation loop
    for t in 1:time_steps
        # Get action from policy
        a, action_info = action_info(policy, b)

        # Store the action
        push!(all_u, a)

        # Step the POMDP
        s, _, _ = POMDPs.gen(pomdp, s, a, rng)

        # Observation 
        z = POMDPs.observation(pomdp, s, a, rng)

        # Update belief
        b = belief_updater.update(pomdp, b, a, z)
        
        if b === nothing
            return nothing
        end
        
        # Extract the mean and covariance from belief
        m = b[1:num_states(pomdp)]
        Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))
        
        # Store estimates
        push!(means, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)]) # 8 x 1
        push!(variances, diagm(b[end-num_sysvars(pomdp) + 1:end])) # 8x8
        push!(all_b, b)

        # Simulate true state (process noise included)
        push!(all_s, s)
    end

    ΣΘΘ = b[end]
    info_dict = Dict(:all_b => all_b, :means => means, :variances => variances, :ΣΘΘ => ΣΘΘ, :all_s => all_s, :all_u => all_u, :true_params => pomdp.true_params)
    return all_b, info_dict
end

