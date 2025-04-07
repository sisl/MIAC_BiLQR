# ==============================================================================
# AirplanePOMDP Definition
# A POMDP model representing the dynamics and observations of an airplane system
# ==============================================================================
mutable struct AirplanePOMDP <: iLQRPOMDP{AbstractVector, AbstractVector, AbstractVector}

    # Cost matrices
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Q_N::Matrix{Float64}
    Λ::Matrix{Float64}

    # Initial belief and goal state
    Σ0::Vector{Float64}
    b0::MvNormal
    s_init::Vector{Float64}
    AB_true::Vector{Float64}
    s_goal::Vector{Float64}

    # Physical constants
    m::Float64
    g::Float64
    l::Float64
    δt::Float64

    # Noise covariance matrices
    W_state_process::Matrix{Float64}
    W_process::Matrix{Float64}
    W_obs::Matrix{Float64}
    W_obs_ekf::Matrix{Float64}

    # Dimensionality
    num_states::Int # belief states 
    num_actions::Int
    num_observations::Int
    num_sysvars::Int

    # Constructor
    function AirplanePOMDP(
        Q::AbstractMatrix{Float64}, R::AbstractMatrix{Float64}, Q_N::AbstractMatrix{Float64}, Λ::AbstractMatrix{Float64},
        m0::Vector{Float64}, Σ0::Vector{Float64}, δt::Float64, m::Float64, g::Float64, s_goal::Vector{Float64},
        W_process::AbstractMatrix{Float64}, W_obs::AbstractMatrix{Float64}, num_truestates::Int, num_actions::Int, num_observations::Int, num_sysvars::Int
    )

        b0 = MvNormal(m0, diagm(Σ0))
        s_init = rand(b0)
        AB_true = s_init[end-num_sysvars+1:end]

        num_states = num_truestates + num_sysvars
        s_goal = [s_goal...; vec(zeros(num_sysvars))...] # goal belief state 

        # Create W_state_process as a diagonal matrix with the first num_states - num_sysvars diagonal elements of W_process
        vector = diag(W_process)  
        sub_vector = vector[1:(num_states - num_sysvars)]  # First 7 elements
        W_state_process = diagm(sub_vector)

        true_params = []
        push!(true_params, AB_true)

        model = new(Q, R, Q_N, Λ, m0, Σ0, b0, s_init, AB_true, s_goal, δt, m, g, W_state_process, W_process, W_obs, 
        num_states, num_actions, num_observations, num_sysvars)

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

        return model

    end
end

# ==============================================================================
# Dynamics and Observations
# Define system dynamics and observation functions for the AirplanePOMDP
# ==============================================================================

"""
    dyn_mean(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)

Compute the mean dynamics update for the airplane system.
"""
function dyn_mean(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)
    # Extract state and parameters
    s_true = s[1:4]
    A = s[5:8]
    B = s[9:end]

    # Define system matrices
    col2 = [0.0, -0.1, -0.5, 0.0]
    col3 = [-9.81, 1.0, -0.1, 1.0]
    col4 = [0.0, 0.0, 0.0, 0.0]
    colB = [0.0, 0.0, 0.0, 0.0]

    # Compute dynamics
    ds = hcat(A, col2, col3, col4) * s_true + hcat(B, colB) * a
    s_new = s_true + p.δt * ds

    # Return updated state
    return vcat(s_new, A, B)
end

"""
    dyn_noise(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)

Return the process noise covariance for the airplane system.
"""
dyn_noise(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process

"""
    obs_mean(p::AirplanePOMDP, sp::AbstractVector)

Return the mean observation for the airplane system.
"""
obs_mean(p::AirplanePOMDP, sp::AbstractVector) = sp[1:4]

"""
    obs_noise(p::AirplanePOMDP, sp::AbstractVector)

Return the observation noise covariance for the airplane system.
"""
obs_noise(p::AirplanePOMDP, sp::AbstractVector) = p.W_obs
