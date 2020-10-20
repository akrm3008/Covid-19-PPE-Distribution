import numpy as np
from plot import plot 
import tensorflow as tf
import datetime
from environment import get_env
from DQN import DQN, run

np.random.seed(1)

demand_points_list = ['Brooklyn', 'Bronx', 'Staten Island','Manhattan', 'Queens']
time_horizon = 20                                                                           # number of days
hospitalisation_mean = [150, 100, 50, 200, 70]
hospitalisation_sd = [30, 20, 10, 10, 20]
hospitalisation_trend = ['up','up','up','up','up']                                                               
initial_inv = [100000, 80000, 85000, 50000, 50000]
#initial_inv_mean = 100000                                                                  # initial inventory mean for simulation
#initial_inv_sd = 5000                                                                      # initial inventory standar deviation for simulation
lams = np.zeros(shape = (len(demand_points_list), time_horizon))                            # arrival rates for hospitalisation for each demand point for different 
for i in range(len(demand_points_list)):
  lams[i] = np.random.normal(hospitalisation_mean[i], hospitalisation_sd[i],  time_horizon)
burn_rate_mean = [6, 6, 6, 6, 6]                                                            # burn rate mean (burn rate is calculated per day per patient)
burn_rate_sd = [2, 2, 2, 2, 2]                                                              # burn rate sd             
death_rate = 0.04                                                                           # Proportion of people hospitalised that die
mean_LOS_death = 7                                                                          # Avergae LOS (Length of Stay) for patients that die (in days)
mean_LOS_discharge = 10                                                                     # Average LOS for patients that get discarhegd (in days)
avg_interarrival_time_supply = 7                                                            # On an average weekly interarrival between supplies 
supply_amount = 10000 
vehicles = 100



# Rendering the environement
env = get_env(demand_points_list, vehicles, time_horizon)
env.set(initial_inv = initial_inv , lams = lams, hospitalisation_trend = hospitalisation_trend, 
                    burn_rate_mean = burn_rate_mean, burn_rate_sd = burn_rate_sd, death_rate = death_rate , mean_LOS_death = mean_LOS_death, 
                    mean_LOS_discharge = mean_LOS_discharge, average_interarrival_time = avg_interarrival_time_supply, supply_amount = supply_amount)



# Simulating states/state variables if no action is taken 
env.actionlesss_simulation()

## Plotting the results of the simulation

# Plotting hospitalisations, deaths, discharges and active cases for each demand point by day
plot.plot_cases(time_horizon, env, demand_points_list)

# Plotting PPE inventory for each demand point by day
plot.plot_inventory(time_horizon, env, demand_points_list)


# Plotting PPE usage for each demand point by day
plot.plot_PPE_usage(time_horizon, env, demand_points_list)

# Plotting PPE usage for each demand point by day
plot.plot_surplus_inventory(time_horizon, env, demand_points_list)

# Reseting the environment
env.reset()

# Generating a random action containing inventory transfers and vehicle trasnfers 
action_inventory = np.random.normal(loc = 100, scale = 10, size = int((1/2)*len(demand_points_list)*(len(demand_points_list) - 1))).round()
action_vehicles = np.random.normal(loc = 8, scale = 3, size = int((1/2)*len(demand_points_list)*(len(demand_points_list) - 1))).round()
action = np.concatenate((action_inventory, action_vehicles), axis = 0)

# Seeing the states and the reward for this action 
state, rewards, done = env.step(action)

## Building the DQN

gamma = 0.99
copy_freq = 5
num_states = env.state.shape[0]
num_actions = len(demand_points_list)*(len(demand_points_list) - 1)
hidden_units = [24]
max_experiences = 100000
min_experiences = 100
batch_size = 32
lr = 0.001

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# Create train and traget networks
TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences,\
                                                min_experiences, batch_size, lr)
TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences,\
                                                min_experiences, batch_size, lr)
N = 50000

total_rewards = np.empty(N)

epsilon = 0.99
decay = 0.9999
min_epsilon = 0.01

for n in range(N):
  epsilon = max(min_epsilon, epsilon * decay)
  total_reward = run(env, TrainNet, TargetNet, epsilon, copy_freq)
  total_rewards[n] = total_reward
  avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
  with summary_writer.as_default():
    tf.summary.scalar('episode reward', total_reward, step=n)
    tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)

    if n % 100 == 0:
        print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, \
                                        "avg reward (last 100):", avg_rewards)
  print("avg reward for last 100 episodes:", avg_rewards)
