using Parameters
using LinearAlgebra
using SparseArrays
using Distributions
include("../BiLQR/ilqr_types.jl")

@with_kw mutable struct XPlanePOMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    
    Q::Matrix{Float16} = 1e-5 * I(12)
    R::Matrix{Float16} = 1e-5 * I(2)
    Q_N::Matrix{Float16} = diagm(vcat(fill(1e-5, 4), fill(0.1, 8))) # 24 variables
    Λ::Matrix{Float16} = diagm(vcat(fill(1e-5, 4), fill(1, 8)))  # diagonalized \Sigma only has 24 variables, 24 x 24 matrix

    # Σ0::Matrix{Float64} = Diagonal(vcat(fill(1e-10, 8), fill(2, 16)))
    Σ0::Vector{Float64} = vcat(fill(1e-4, 4), fill(10.0, 8))
    b0::MvNormal = MvNormal(
        vcat([1.0, 0.5, 0.1, 0.05], 
             [-0.05, 0, 0, 0], # A unknowns 
             [0, 1, 0, 0]), # B unknowns
        diagm(Σ0))
    s_init::Vector{Float64} = rand(b0)
    # A_true::Matrix{Float64} = Diagonal(s_init[8:15])
    # B_true::Matrix{Float64} = hcat(s_init[16:end], ones(8, 2))
    AB_true::Vector{Float64} = vcat(s_init[5:end])

    # move to new x position, keep same height, and same angle of attack 
    s_goal::Vector{Float64} = vcat([100, 0, 0, 0], s_init[5:end], vec(zeros(12))...)
    
    # mechanics
    m::Float16 = 6500.0
    g::Float16 = 9.81
    l::Float16 = 1.0
    δt::Float16 = 0.1

    # noise
    W_state_process::Matrix{Float16} = diagm([10, 10, 0.1, 0.1])
    W_process::Matrix{Float16} = diagm(vcat([10, 10, 0.1, 0.1, 
                                            0,  0, 0,  0, 
                                            0,  0, 0,  0])) 
    # W_obs::Matrix{Float16} = 1e-2 * Matrix{Float16}(I, 4, 4)
    # W_obs_ekf::Matrix{Float16} = 1e-2 * Matrix{Float16}(I, 4, 4)
    W_obs::Matrix{Float16} = 1e-2*I(3)
    W_obs_ekf::Matrix{Float16} = 1e-2*I(3)
end

function dyn_mean(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector)
    # Ax + Bu = x_new, A' = A, B' = B

    s_true = s[1:4]
    A = s[5:8]
    B = s[9:end]

    # Define the additional columns
    col2 = [0.0, -0.1, -0.5, 0.0]
    col3 = [-9.81, 1.0, -0.1, 1.0]
    col4 = [0.0, 0.0, 0.0, 0.0]

    colB = [0.0, 0.0, 0.0, 0.0]

    # Concatenate the columns to form ds
    ds = hcat(A, col2, col3, col4) * s_true + hcat(B, colB) * a

    s_new = s_true + p.δt * ds

    return vcat(s_new, A, B)
end

dyn_noise(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::XPlanePOMDP, sp::AbstractVector) = sp[1:3]
obs_noise(p::XPlanePOMDP, sp::AbstractVector) = p.W_obs
num_states(p::XPlanePOMDP) = 4 + 4 + 4 
num_actions(p::XPlanePOMDP) = 2
num_observations(p::XPlanePOMDP) = 3
num_sysvars(p::XPlanePOMDP) = 4 + 4