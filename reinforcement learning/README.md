## Reinforcement Learning Algorithm
```
function Base.run(policy, env, stop_condition, hook)
    hook(PRE_EXPERIMENT_STAGE, env)
    while true
        reset!(env)
        policy(PRE_EPISODE_STAGE, env) # pop out the latest dummy state and action 
 pair from trajectory
        hook(PRE_EPISODE_STAGE, env)
        while !is_terminated(env)
            action = policy(env)       # explorer returns probability distribution 
of actions on rewards (returned from approximator)
            policy(PRE_ACT_STAGE, env, action) # policy gets updated & push the 
        current state and action into trajectory
            hook(PRE_ACT_STAGE, env, action)
            env(action) 			  # environment changes state given action
            policy(POST_ACT_STAGE, env) # query reward and termination signal from 
the environment and push them into the      trajectory
            hook(POST_ACT_STAGE, env)
            stop_condition(policy, env) && return
        end
        policy(POST_EPISODE_STAGE, env) # push the state at the end of an episode and a dummy action into the trajectory
        hook(POST_EPISODE_STAGE, env)
    end
end

```
