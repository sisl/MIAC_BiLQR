using JLD2
using POMDPs
include("../Cartpole/cartpole_sysid_tests.jl")

# Initialize dictionaries to store outputs for each seed
all_b = Dict{Int, Vector{Vector{Float64}}}()
all_mp_estimates = Dict{Int, Vector{Float64}}()
all_mp_variances = Dict{Int, Vector{Float64}}()
all_ΣΘΘ = Dict{Int, Float64}()
all_s = Dict{Int, Vector{Vector{Float64}}}()
all_u = Dict{Int, Vector{Vector{Float64}}}()
all_mp_true = Dict{Int, Float64}()

method = "bilqr"
num_seeds = 200

# JLD2 file to save results
jld2_file = "$(method)_cartpolefull_sysid_results.jld2"

# Load existing results if available
if isfile(jld2_file)
    @load jld2_file all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true
end

function get_policy(method::String, pomdp::POMDP)
    if method == "bilqr"
        return BiLQRPolicy(pomdp, N=10, eps=1e-3, max_iters=100)
    elseif method == "mpc"
        return MPCPolicy(pomdp, horizon=10)
    elseif method == "random"
        return RandomPolicy([(-1.0, 1.0), (-5.0, 5.0)])  # Example ranges
    elseif method == "regression"
        return RegressionPolicy(pomdp)
    elseif method == "mpcreg"
        return MPCRegressionPolicy(pomdp, horizon=10)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

# Define a helper function to run a single seed's experiment using `stepthrough`
function run_experiment(seed, pomdp, policy, num_steps)
    Random.seed!(seed)
    
    # Initialize storage
    mp_estimates = Float64[]
    mp_variances = Float64[]
    all_s = Vector{Vector{Float64}}()
    all_b = Vector{Vector{Float64}}()
    all_u = Vector{Vector{Float64}}()

    # Get the initial belief
    b0 = initialstate_distribution(pomdp).support[1]

    # Run simulation via stepthrough
    for (b, a, _, _) in simulate(pomdp, policy, "b,a,z,r", belief=b0)
        # Store data
        push!(all_b, b)
        push!(all_u, a)
        push!(mp_estimates, b[num_states(pomdp)])
        push!(mp_variances, b[end - 1])

        # True state for visualization
        s_true = b[1:num_states(pomdp)]
        push!(all_s, s_true)

        # Stop after `num_steps`
        if length(all_b) >= num_steps
            break
        end
    end

    ΣΘΘ = all_b[end][end]
    return all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, pomdp.mp_true
end

# Run experiments for all seeds
for seed in 1:num_seeds
    println("Seed: ", seed)

    # Skip seeds already processed
    if haskey(all_b, seed)
        println("Seed $seed already processed. Skipping.")
        continue
    end

    # Initialize the POMDP
    pomdp = CartpoleMDP()

    # Get the policy for the specified method
    try
        policy = get_policy(method, pomdp)
    catch e
        println("Failed to initialize policy for method '$method': $e")
        continue
    end

    # Run the experiment
    try
        b_seed, mp_estimates_seed, mp_variances_seed, ΣΘΘ_seed, s_seed, u_seed, mp_true_seed = run_experiment(seed, pomdp, policy, 50)

        # Store results
        all_b[seed] = b_seed
        all_mp_estimates[seed] = mp_estimates_seed
        all_mp_variances[seed] = mp_variances_seed
        all_ΣΘΘ[seed] = ΣΘΘ_seed
        all_s[seed] = s_seed
        all_u[seed] = u_seed
        all_mp_true[seed] = mp_true_seed

        # Save results to JLD2 file
        @save jld2_file all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

    catch e
        println("Seed $seed failed with error: $e. Continuing to next seed.")
        continue
    end
end
