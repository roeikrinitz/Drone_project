import math
import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
        obs_dict=None,
        render_object_shrinking = True
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }


        self._agent_selector = agent_selector(self.agents)

        self.render_object_shrinking = render_object_shrinking

        # set agents observation radiuses
        self.obs_radiuses = []
        total_num_of_agents = len(self.world.agents)
        agent_count= 0
        for obs_radius in obs_dict:
            curr_num_of_agents = obs_dict[obs_radius]
            if curr_num_of_agents > total_num_of_agents - agent_count:
                curr_num_of_agents = total_num_of_agents - agent_count

            self.obs_radiuses.extend([obs_radius] * curr_num_of_agents)

            if curr_num_of_agents + agent_count >= total_num_of_agents:
                break
            agent_count += curr_num_of_agents

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + (self.world.lamp_flag+self.world.height_flag+1)    # updated from +1 to +4 (for lamp, height and color action)
                if self.world.num_of_possible_colors_for_agent > 1:
                    space_dim += 1

            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            if self._index_map[agent.name] >= len(self.obs_radiuses):
                radius = 2
            else:
                radius = self.obs_radiuses[self._index_map[agent.name]]
            obs_dim = len(self.scenario.observation(agent, self.world, radius))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        if self._index_map[agent] >= len(self.obs_radiuses):
            radius = 2
        else:
            radius = self.obs_radiuses[self._index_map[agent]]
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world, radius
        )

    def state(self):    #   States are not updated according to the observation radius of each agent
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + (self.world.lamp_flag + self.world.height_flag+1)  # updated from +1 to +4 (for lamp, height and color action)
                if self.world.num_of_possible_colors_for_agent > 1:
                    mdim += 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

        for agent1 in self.world.agents:
            for agent2 in self.world.agents:
                if agent1 == agent2:
                    continue
                if self.scenario.is_collision(agent1, agent2)\
                        and((agent2.adversary and not agent1.adversary) or(agent1.adversary and not agent2.adversary))\
                        and not self.world.dead_list[agent1.name] and not self.world.dead_list[agent2.name]:
                    for x in [agent1,agent2]:
                        self.world.dead_list[x.name] = True
                        x.max_speed = 0
                        x.accel = 0
                        x.size = 0

        for agent in self.world.agents:
            for landmark in self.world.landmarks:
                if agent == landmark:
                    continue
                if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) <= landmark.size and not self.world.dead_list[agent.name]:
                    self.world.shadow_list[agent.name] = True
                else:
                    self.world.shadow_list[agent.name] = False

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            agent.action.lamp_change = False
            agent.action.height_change = False
            agent.action.color_change = False
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
                if action[0] in self.world.action_dict and agent.adversary:
                    self.world.action_dict[action[0]][0](agent)
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # Get the next agent, who isn't dead:
        current_idx = self._index_map[self.agent_selection]
        # actual_agent = self.world.agents[current_idx]
        # while actual_agent.name in self.world.dead_list and (self.world.dead_list[actual_agent.name]):
        #     next_idx = (current_idx + 1) % self.num_agents
        #     if next_idx == 0:
        #         break
        #
        #     self.agent_selection = self._agent_selector.next()
        #     current_idx = self._index_map[self.agent_selection]
        #     actual_agent = self.world.agents[current_idx]


        cur_agent = self.agent_selection

        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [(1,0)]
        for entity in self.world.entities:
            if entity.name in self.world.dead_list and (not self.world.dead_list[entity.name]):
                all_poses.append(entity.state.p_pos)
        cam_range = np.max(np.abs(np.array(all_poses)))

        if self.render_object_shrinking:
            relative_size_reduction = min(math.sqrt(cam_range), 12)
        # 12 is a constant factor, so the objects won't disappear relative to the cam range
        else:
            relative_size_reduction = 1

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(reversed(self.world.entities)):
            if entity.name in self.world.dead_list and self.world.dead_list[entity.name]:
                continue
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            if entity.movable and entity.adversary:
                curr_color = self.world.colors_for_agent[entity.state.color_index]
            else:
                curr_color = entity.color
            pygame.draw.circle(
                self.screen, curr_color * 200, (x, y), entity.size * 350 * (1 / relative_size_reduction)
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350 * (1 / relative_size_reduction), 1
            )  # borders
            if entity.state.lamp:
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), entity.size * 450 * (1 / relative_size_reduction),1
                )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            if entity.state.height:
                pygame.draw.circle(
                    self.screen, entity.color * 100, (x, y), entity.size * 100 * (1 / relative_size_reduction)
                ) # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            #pygame.display.quit()
            self.renderOn = False
