# using BiLQR 
# using ForwardDiff

struct EKFUpdater <: POMDPs.Updater
    pomdp::iLQRPOMDP
end

function m_sigma_update(pomdp, x::AbstractVector, Σ::AbstractMatrix, u::AbstractVector, z::AbstractVector)
    
    Ct = ForwardDiff.jacobian(x -> obs_mean(pomdp,x, u), x)

    y = z - obs_mean(pomdp, x, u)
    S = Ct * Σ * Ct' + obs_noise(pomdp, x, u)

    # if any(isnan, S) || abs(det(S)) < 1e-12
    #     println("BiLQR S is nan, next seed...")
    #     return nothing
    # end

    # check if this line needs obs noise 
    K = Σ * Ct' * inv(S)

    # Update the mean estimate
    m_new = x + K * y

    # Update the covariance estimate
    Σ_new = (I - K * Ct) * Σ

    return m_new, Σ_new

end

function update(ekf::EKFUpdater, b, a, z)
    
    pomdp = ekf.pomdp
    m, Σ = b[1:pomdp.num_states], reshape(b[pomdp.num_states + 1:end], pomdp.num_states, pomdp.num_states)

    m_pred = dyn_mean(pomdp, m, a)

    At = ForwardDiff.jacobian(x -> dyn_mean(pomdp, x, a), m_pred)
    Σ_pred = At * Σ * At' + dyn_noise(pomdp, m_pred, a)

    m_new, Σ_new = m_sigma_update(pomdp, m_pred, Σ_pred, a, z)

    # if Σ_new === nothing
    #     return nothing
    # end

    return [m..., vec(Σ)...]

end