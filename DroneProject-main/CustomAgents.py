from pettingzoo.mpe import simple_tag_v3
import numpy as np

def smart_green_agent(observation, env,agent):
    min_agent_norm = float('inf')
    min_agent_loc = [0, 0]
    action = env.action_space(agent).sample()  # this is where you would insert your policy
    found_enemy = False
    for item in observation[3:]:
        target_adversery, target_location, target_color = item
        if target_adversery:
            found_enemy = True
            if np.linalg.norm(target_location) < min_agent_norm:
                min_agent_norm = np.linalg.norm(target_location)
                min_agent_loc = target_location
    if abs(min_agent_loc[0]) <= abs(min_agent_loc[1]):
        if min_agent_loc[0] <= 0:
            action = 1
        else:
            action = 2
    else:
        if min_agent_loc[1] <= 0:
            action = 3
        else:
            action = 4
    if found_enemy == False:
        action = env.action_space(agent).sample()
    return action

def stupidAgent(observation, env,agent):
    min_agent_norm = float('inf')
    min_agent_loc = [0, 0]
    action = env.action_space(agent).sample()  # this is where you would insert your policy
    found_enemy = False
    for item in observation[3:]:
        target_adversery, target_location, target_color = item
        if not target_adversery:
            found_enemy = True
            if np.linalg.norm(target_location) < min_agent_norm:
                min_agent_norm = np.linalg.norm(target_location)
                min_agent_loc = target_location
    if abs(min_agent_loc[0]) > abs(min_agent_loc[1]):
        if min_agent_loc[0] <= 0:
            action = 1
        else:
            action = 2
    else:
        if min_agent_loc[1] <= 0:
            action = 3
        else:
            action = 4
    if found_enemy == False:
        action = env.action_space(agent).sample()
    return action

def greedyAgent(observation,env,agent):
    min_agent_norm = float('inf')
    min_agent_loc = [0, 0]
    action = env.action_space(agent).sample()  # this is where you would insert your policy
    found_enemy = False
    for item in observation[3:]:
        target_adversery, target_location, target_color = item
        if not target_adversery:
            found_enemy = True
            if np.linalg.norm(target_location) < min_agent_norm:
                min_agent_norm = np.linalg.norm(target_location)
                min_agent_loc = target_location
    if abs(min_agent_loc[0]) > abs(min_agent_loc[1]):
        if min_agent_loc[0] <= 0:
            action = 2
        else:
            action = 1
    else:
        if min_agent_loc[1] <= 0:
            action = 4
        else:
            action = 3
    if found_enemy == False:
        action = env.action_space(agent).sample()
    return action


def randomAgent(observation,env,agent):
    return env.action_space(agent).sample()


def staticAgent(observation,env,agent):
    return 0

