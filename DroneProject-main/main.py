# Importing the CustomAgents module
from CustomAgents import *
from Auxiliry import *


# The simulator is a drone environment simulation where there are enemies represented as green circles
# and drones represented as red circles. It aims to create an interactive environment where drones and
# enemies can navigate, interact, and potentially engage in strategic behaviors.
# The drones and enemies can move in various directions within the environment, allowing for dynamic
# movement patterns and interactions.
# The drones have additional capabilities, such as the ability to change colors and reach higher ground,
# providing them with increased versatility and strategic options.
# In addition to the drones and enemies, the environment features shadow zones, which are areas where both
# drones and enemies can enter. These shadow zones introduce an additional element to the simulation and
# can potentially affect the behavior and interactions of the entities within them.
# Overall, the simulator aims to provide a realistic and immersive simulation of a drone-based scenario,
# allowing for the exploration and analysis of various drone and enemy interactions within the simulated
# environment.




# obs_dict: A dictionary of (key=radius, value=num_of_agents_with_radius).
# If the total number of agent in obs_dict is greater than the number of good agents -
#   we ignore the last radiuses in the dict.
# If the total number of agent in obs_dict is smaller than the number of good agents -
#   we assume the remaining agents see all the map

# Note: The radiuses values are in the range [0, 2]
my_obs_dict = {20: 2, 0.00001: 1}

# Factors dictionary that contains some factors used in the environment -
factors = {"height_other_factor": 1}

# There are 2 ways to initialize the environment:

# 1. Empty Constructor (Parameters from Setting File)
#    - Initialize the environment using an empty constructor.
#    - Read parameters from a settings file with predefined values.
#    - Allows easy configuration without modifying the code directly.
env = simple_tag_v3.env()

# 2. Parameter Constructor
#    - Initialize the environment by directly passing parameters to the constructor.
#    - Parameters include render mode, agent and adversary counts, obstacle count, maximum cycles,
#      observation and factor dictionaries, and possible agent colors.
#    - Provides flexibility and customization for specific requirements or dynamic conditions.

env = simple_tag_v3.env(
    render_mode='human',
    num_good=5,
    num_adversaries=5,
    num_obstacles=3,
    max_cycles=1000,
    obs_dict=my_obs_dict,
    factor_dict=factors,
    num_of_possible_colors_for_agent=3,
    lamp_flag=True,
    height_flag=True,
    landmark_colide=False,
)
# Prints the actions the agent can perform
print_action_dict(env)

# Reset the environment and get initial observation, reward, termination, truncation, and info
env.reset()
observation, reward, termination, truncation, info = env.last()

# The observation is a list of positions the first agent can observe (according to the radius).

run_env(env,greedyAgent,staticAgent)

# Close the environment
env.close()

# Reset the environment and get initial observation, reward, termination, truncation, and info
env.reset()
observation, reward, termination, truncation, info = env.last()

# The observation is a list of positions the first agent can observe (according to the radius).
