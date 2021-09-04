# Reinforcement Learning Algorithm

## Common Terms Explained
A policy `(policy::AbstractPolicy)` takes a look at the environment (observation) and yields an action, while an environment `(env::AbstractEnv)` will modify its internal state once received an action. 
We stop the workflow early after a number of steps/episodes or when the policy or environment meet some specific condition. A stop_condition which examines policy and env after each step to control the workflow.

### Environment: 
Common environment interfaces:
```
reset!(env)     # resets the env to an initial state
actions(env)    # returns the set of all possible actions for the environment
observe(env)    # returns an observation
act!(env, a)    # steps the environment forward and returns a reward
terminated(env) # returns true or false indicating whether the environment has finished
```
### Agent: 
If a policy needs to be updated during interactions with the environment, it needs a buffer to collect some necessary information and then use it to update its strategy at some time. A special policy named Agent is used to update a policy. An agent is simply a combination of any policy to be updated and a corresponding experience replay buffer `(trajectory::AbstractTrajectory)`.

### Trajectory:
A trajectory contains four traces: state, action, reward and terminal. Each trace simply uses a Vector as the container. By default, the trajectory is assumed to be of the SARTSA format (State, Action, Reward, Termination, next-State, next-Action).

### Hook:
We may want to perform some actions during the interactions between the policy and the environment. For example, collecting the total reward of each episode, logging the loss of each update, saving the policy periodically, modifying some hyperparameters on the fly and so on. A general callback `(hook::AbstractHook)` is introduced.


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
