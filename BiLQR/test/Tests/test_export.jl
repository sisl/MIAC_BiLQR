using BiLQRExperiments # exports pomdps, policies, and simulate

# BiLQR - fully-observable cart-pole SysID

pomdp = Cartpole(Lambda = high, Q = 0, R = 0)
policy = BiLQR(…..)
belief_updater = EKF(pomdp)
results = simulate(pomdp, policy, belief_updater, n_seeds=10)
# save results

# MPC+Regression - SysId - partially observable xplane
pomdp = Xplane(observation_noise = different than default to reflect partially observable)
policy = MPC(…)
belief_updater = Regression(…)
Results = simulate(pomdp, policy, belief_updater, n_seeds=10)
# save results

# All fullobs-Cartpole MIAC experiments at once
pomdp = Cartpole(Lambda = high, Q=high, R=high, obs_noise = low)
policies_and_updaters = [(Random, EKF),
                        (MPC, EKF),
                        (MPC, Regression),
                        (BiLQR, EKF) ] 
for (policy, updater) in policies_and_updaters
    results = simulate(pomdp, policy, belief_updater, n_seeds)
    # save results
end 
