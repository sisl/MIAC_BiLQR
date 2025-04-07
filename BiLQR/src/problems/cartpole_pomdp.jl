
# ==============================================================================
# CartpoleMDP Definition
# A POMDP model representing the dynamics and observations of a cart-pole system
# ==============================================================================

mutable struct CartpoleMDP <: iLQRPOMDP{AbstractVector, AbstractVector, AbstractVector}

    # Cost matrices
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Q_N::Matrix{Float64}
    Λ::Matrix{Float64}

    # Initial belief and goal state
    m0::Vector{Float64}
    Σ0::Vector{Float64}
    s_init::Vector{Float64}
    mp_true::Float64
    s_goal::Vector{Float64}

    # Physical parameters
    δt::Float64
    mc::Float64
    g::Float64
    l::Float64

    # Noise covariance matrices
    W_state_process::Matrix{Float64}
    W_process::Matrix{Float64}
    W_obs::Matrix{Float64}
    W_obs_ekf::Matrix{Float64}

    # Dimensionality
    num_states::Int # number of belief states 
    num_actions::Int
    num_observations::Int
    num_sysvars::Int

    true_params::Vector{Float64}

    function CartpoleMDP(
        Q::AbstractMatrix{Float64}, R::AbstractMatrix{Float64}, Q_N::AbstractMatrix{Float64}, Λ::AbstractMatrix{Float64},
        m0::Vector{Float64}, Σ0::Vector{Float64}, δt::Float64, mc::Float64, g::Float64, l::Float64,
        W_process::AbstractMatrix{Float64}, W_obs::AbstractMatrix{Float64}, num_truestates::Int, num_actions::Int, num_observations::Int, num_sysvars::Int
    )
        b0 = MvNormal(m0, diagm(Σ0))
        s_init = rand(b0)
        s_init[end] = abs(s_init[end])  # Ensure mass is positive
        mp_true = s_init[end]
        s_goal = [s_init...; vec(zeros(5,5))...]

        # Create W_state_process as a diagonal matrix with the first num_states - num_sysvars diagonal elements of W_process
        vector = diag(W_process)  
        sub_vector = vector[1:(num_states - num_sysvars)]  # First 7 elements
        W_state_process = diagm(sub_vector)

        true_params = []
        push!(true_params, mp_true)

        num_states = num_truestates + num_sysvars

        model = new(Q, R, Q_N, Λ, m0, Σ0, b0, s_init, mp_true, s_goal, δt, mc, g, l, W_state_process, W_process, W_obs, 
                    num_states, num_actions, num_observations, num_sysvars)
        
        return model
    
        
        # Dimension validation
        @assert size(Q) == (num_states(model), num_states(model)) "Q matrix must be $(num_states(model))x$(num_states(model))"
        @assert size(Q_N) == (num_states(model), num_states(model)) "Q_N matrix must be $(num_states(model))x$(num_states(model))"
        @assert size(R) == (num_actions(model), num_actions(model)) "R matrix must be $(num_actions(model))x$(num_actions(model))"
        @assert size(Λ) == (num_states(model), num_states(model)) "Λ matrix must be $(num_states(model))x$(num_states(model))"
        @assert length(m0) == num_states(model) "Initial mean vector (m0) must have length $(num_states(model))"
        @assert size(Σ0) == (num_states(model), num_states(model)) "Initial covariance matrix (Σ0) must be $(num_states(model))x$(num_states(model))"
        @assert size(W_state_process) == (num_states(model), num_states(model)) "W_state_process must be $(num_states(model))x$(num_states(model))"
        @assert size(W_process) == (num_states(model), num_states(model)) "W_process must be $(num_states(model))x$(num_states(model))"
        @assert size(W_obs) == (num_observations(model), num_observations(model)) "W_obs must be $(num_observations(model))x$(num_observations(model))"
        @assert size(W_obs_ekf) == (num_observations(model), num_observations(model)) "W_obs_ekf must be $(num_observations(model))x$(num_observations(model))"
        
     end 
end

# ==============================================================================
# Dynamics and Observations
# Define system dynamics and observation functions for the CartpoleMDP
# ==============================================================================

"""
    dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)

Compute the mean dynamics update for the cart-pole system.
"""
function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
    # Extract state variables
    x, θ, dx, dθ, mp = s
    sinθ, cosθ = sin(θ), cos(θ)
    h = p.mc + mp * (sinθ^2)

    # Compute state derivatives
    ds = [
        dx,  # x_dot
        dθ,  # θ_dot
        (mp * sinθ * (p.l * (dθ^2) + p.g * cosθ) + a[1]) / h,  # x_ddot
        -((p.mc + mp) * p.g * sinθ + mp * p.l * (dθ^2) * sinθ * cosθ + a[1] * cosθ) / (h * p.l),  # θ_ddot
        0.0  # mass remains constant
    ]

    # Update state using Euler integration
    s_new = s + p.δt * ds

    return s_new
end

"""
    dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)

Return the process noise covariance for the cart-pole system.
"""
dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process

"""
    obs_mean(p::CartpoleMDP, sp::AbstractVector)

Return the mean observation for the cart-pole system.
"""
obs_mean(p::CartpoleMDP, sp::AbstractVector, a::AbstractVector) = sp[1:num_observations(p)]

"""
    obs_noise(p::CartpoleMDP, sp::AbstractVector)

Return the observation noise covariance for the cart-pole system.
"""
obs_noise(p::CartpoleMDP, sp::AbstractVector, a::AbstractVector) = p.W_obs