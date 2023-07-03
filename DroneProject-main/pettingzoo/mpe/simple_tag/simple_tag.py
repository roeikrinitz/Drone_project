# noqa

"""
# Simple Tag

```{figure} mpe_simple_tag.gif
:width: 140px
:name: simple_tag
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_tag_v3`                 |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
from pettingzoo.mpe.simple_tag.settings import *
import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn



class raw_env(SimpleEnv, EzPickle):
    def __init__(
    self,
    obs_dict=obs_dict,
    render_mode =render_mode,
    num_good = num_good,
    num_adversaries = num_adversaries,
    num_obstacles = num_obstacles,
    max_cycles = max_cycles,
    num_of_possible_colors_for_agent = num_of_possible_colors_for_agent,
    render_object_shrinking = render_object_shrinking,
    factor_dict = {},
    reward_dict = {},
    lamp_flag=False,
    height_flag=False,
    landmark_colide = False,


    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=False,
            render_mode=render_mode,
        )
        scenario = Scenario()
        factor_dict = factor_dict_reset(factor_dict)
        reward_dict = reward_dict_reset(reward_dict)
        world = scenario.make_world(num_good, num_adversaries, num_obstacles,
                                    factor_dict,reward_dict,num_of_possible_colors_for_agent,lamp_flag,height_flag,landmark_colide)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=False,
            obs_dict=obs_dict,
            render_object_shrinking = render_object_shrinking,
        )
        self.metadata["name"] = "simple_tag_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

def factor_dict_reset(factor_dict):
    if not ("shadow_interference_factor" in factor_dict):
        factor_dict["shadow_interference_factor"] = factor_dict_default["shadow_interference_factor"]
    if not ("lamp_improvement_factor" in factor_dict):
        factor_dict["lamp_improvement_factor"] = factor_dict_default["lamp_improvement_factor"]
    if not ("other_shadow_interference_factor" in factor_dict):
        factor_dict["other_shadow_interference_factor"] = factor_dict_default["other_shadow_interference_factor"]
    if not ("height_adversary_factor" in factor_dict):
        factor_dict["height_adversary_factor"] = factor_dict_default["height_adversary_factor"]
    if not ("height_non_adversary_factor" in factor_dict):
        factor_dict["height_non_adversary_factor"] = factor_dict_default["height_non_adversary_factor"]
    if not ("height_other_factor" in factor_dict):
        factor_dict["height_other_factor"] = factor_dict_default["height_other_factor"]
    if not ("light_in_shadow_factor" in factor_dict):
        factor_dict["light_in_shadow_factor"] = factor_dict_default["light_in_shadow_factor"]
    return factor_dict

def reward_dict_reset(reward_dict):
    if not ("drone_collision" in reward_dict):
        reward_dict["drone_collision"] = reward_dict_default["drone_collision"]
    if not ("drone_in_shadow_lamp_on" in reward_dict):
        reward_dict["drone_in_shadow_lamp_on"] = reward_dict_default["drone_in_shadow_lamp_on"]
    if not ("drone_in_shadow_lamp_off" in reward_dict):
        reward_dict["drone_in_shadow_lamp_off"] = reward_dict_default["drone_in_shadow_lamp_off"]
    if not ("drone_lamp_active" in reward_dict):
        reward_dict["drone_lamp_active"] = reward_dict_default["drone_lamp_active"]
    if not ("drone_lamp_off" in reward_dict):
        reward_dict["drone_lamp_off"] = reward_dict_default["drone_lamp_off"]
    if not ("drone_turn_lamp_on" in reward_dict):
        reward_dict["drone_turn_lamp_on"] = reward_dict_default["drone_turn_lamp_on"]
    if not ("drone_in_height" in reward_dict):
        reward_dict["drone_in_height"] = reward_dict_default["drone_in_height"]
    if not ("drone_height_change" in reward_dict):
        reward_dict["drone_height_change"] = reward_dict_default["drone_height_change"]
    if not ("drone_color_change" in reward_dict):
        reward_dict["drone_color_change"] = reward_dict_default["drone_color_change"]
    if not ("parasite_collision" in reward_dict):
        reward_dict["parasite_collision"] = reward_dict_default["parasite_collision"]
    return reward_dict



class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2,
                   factor_dict = {}, reward_dict = {},num_of_possible_colors_for_agent=0,lamp_flag=False,height_flag=False,landmark_colide=False):
        world = World()
        # set any world properties first
        world.factor_dict = factor_dict
        world.reward_dict = reward_dict
        if num_of_possible_colors_for_agent <= 0:
            num_of_possible_colors_for_agent = 1
        if num_of_possible_colors_for_agent > 5:
            num_of_possible_colors_for_agent = 5
        world.num_of_possible_colors_for_agent = num_of_possible_colors_for_agent
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        world.lamp_flag = lamp_flag
        world.height_flag = height_flag
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.dead_list = {agent.name: False for agent in world.agents}
        world.shadow_list = {agent.name: False for agent in world.agents}
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmark_colide = landmark_colide
        world.landmarks = [Landmark(world.landmark_colide) for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i

            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        action_counter=5
        if world.lamp_flag:
            world.action_dict[action_counter]=(world.lamp_action_dic,"lamp action")
            action_counter+=1
        if world.height_flag:
            world.action_dict[action_counter]=(world.height_action_dic,"get height action")
            action_counter+=1
        if world.num_of_possible_colors_for_agent > 1:
            world.action_dict[action_counter]=(world.color_action_dic,"change color action")

        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.direction = np_random.uniform(0, 360)
        world.dead_list = {agent.name: False for agent in world.agents}
        world.shadow_list = {agent.name: False for agent in world.agents}
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        if world.dead_list[agent.name]:
            return 0
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                if world.dead_list[adv.name]:
                    continue
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if world.dead_list[a.name]:
                    continue
                if self.is_collision(a, agent):
                    rew -= world.reward_dict["parasite_collision"]


        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        if world.dead_list[agent.name]:
            return 0
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                if world.dead_list[adv.name]:
                    continue
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                if world.dead_list[ag.name]:
                    continue
                for adv in adversaries:
                    if world.dead_list[adv.name]:
                        continue
                    if self.is_collision(ag, adv):
                        rew += world.reward_dict["drone_collision"]

        if world.shadow_list[agent.name]:
            if agent.state.lamp:
                rew += world.reward_dict["drone_in_shadow_lamp_on"]
            else:
                rew += world.reward_dict["drone_in_shadow_lamp_off"]

        if agent.state.lamp:
            rew += world.reward_dict["drone_lamp_active"]

        if agent.action.lamp_change and agent.state.lamp:
            rew += world.reward_dict["drone_lamp_off"]

        if agent.action.lamp_change and not agent.state.lamp:
            rew += world.reward_dict["drone_turn_lamp_on"]

        if agent.state.height:
            rew += world.reward_dict["drone_in_height"]

        if agent.action.height_change:
            rew += world.reward_dict["drone_height_change"]

        if agent.action.color_change:
            rew += world.reward_dict["drone_color_change"]

        return rew

    def observation(self, agent, world, obs_radius = 2):
        # get positions of all entities in this agent's reference frame, if the entity is in the agents observation radius

        #if (agent.name in world.dead_list) and (world.dead_list[agent.name]):
        #     return []

        agents_pos = [agent.state.lamp, agent.adversary,agent.state.color_index]
        obs_improvement_factor = 1.0

        # impair vision if inside of a landmark
        for landmark in world.landmarks:
            if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) <= landmark.size:
                obs_improvement_factor = world.factor_dict["shadow_interference_factor"]
                break

        # get vision of all other agents in sight (landmark doesn't count as an agent)
        for other in world.agents:
            if other == agent or world.dead_list[other.name]:
                continue
            curr_obs_improvement_factor = obs_improvement_factor
            relative_distance = agent.state.p_pos - other.state.p_pos

            # lamp and shadow updates
            if agent.state.lamp: # improved vision by lamp assistance
                curr_obs_improvement_factor *= world.factor_dict["lamp_improvement_factor"]

            if world.shadow_list[other.name] and other.state.lamp == False:
                curr_obs_improvement_factor *= world.factor_dict["other_shadow_interference_factor"]

            if world.shadow_list[other.name] and other.state.lamp == True:
                curr_obs_improvement_factor *= world.factor_dict["light_in_shadow_factor"]

            # height updates
            if agent.state.height:
                if other.adversary:
                    curr_obs_improvement_factor *= world.factor_dict["height_adversary_factor"]
                else:
                    curr_obs_improvement_factor *= world.factor_dict["height_non_adversary_factor"]
            elif other.state.height:
                curr_obs_improvement_factor *= world.factor_dict["height_other_factor"]


            if np.linalg.norm(relative_distance) <= obs_radius * curr_obs_improvement_factor:
                agents_pos.append((other.adversary, relative_distance,other.state.color_index))


        return agents_pos
