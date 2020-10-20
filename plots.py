# Creating a class for visualising the simualtions 

import matplotlib.pyplot as plt
import seaborn as sns

# A class which will contain all kinds of methods for visulaisation
class plot():

# Method for plotting hospitalisations, deaths, discharges and active cases by days for different demand points
  def plot_cases(time_horizon, env, demand_points_list):
    
    x = [i for i in range(time_horizon)]

    for i in range(len(env.demand_points)):
      plt.figure()
      plt.plot(x, env.demand_points[i].hospitalisations, label = 'Hospitalisations')
      plt.plot(x, env.demand_points[i].deaths, label = 'Deaths')
      plt.plot(x, env.demand_points[i].discharges, label = 'Discharges')
      plt.plot(x, env.demand_points[i].active_cases, label = 'Active cases')
      plt.xlabel('Days')
      plt.legend()
      plt.title( demand_points_list[i] + 'cases data by days')

# Method for plotting inventory by days for different demand points 
  def plot_inventory(time_horizon, env, demand_points_list):
    x = [i for i in range(time_horizon)]

    plt.figure()
    plt.plot(x, env.demand_points[0].inventory, label = demand_points_list[0])
    plt.plot(x, env.demand_points[1].inventory, label = demand_points_list[1])
    plt.plot(x, env.demand_points[2].inventory, label = demand_points_list[2])
    plt.plot(x, env.demand_points[3].inventory, label = demand_points_list[3])
    plt.plot(x, env.demand_points[4].inventory, label = demand_points_list[4])
    plt.xlabel('Days')
    plt.ylabel('Inventory')
    plt.legend()
    plt.title('PPE inventory data by days')


# Method for plotting PPE_usage by days for different demand points 
  def plot_PPE_usage(time_horizon, env, demand_points_list):
    x = [i for i in range(time_horizon)]

    plt.figure()
    plt.plot(x, env.demand_points[0].PPE_usage, label = demand_points_list[0])
    plt.plot(x, env.demand_points[1].PPE_usage, label = demand_points_list[1])
    plt.plot(x, env.demand_points[2].PPE_usage, label = demand_points_list[2])
    plt.plot(x, env.demand_points[3].PPE_usage, label = demand_points_list[3])
    plt.plot(x, env.demand_points[4].PPE_usage, label = demand_points_list[4])
    plt.xlabel('Days')
    plt.ylabel('PPE Usage')
    plt.legend()
    plt.title('PPE usage data by days')



# Method for plotting PPE_usage by days for different demand points 
  def plot_surplus_inventory(time_horizon, env, demand_points_list):
    x = [i for i in range(time_horizon)]

    plt.figure()
    plt.plot(x, env.demand_points[0].surplus_inventory, label = demand_points_list[0])
    plt.plot(x, env.demand_points[1].surplus_inventory, label = demand_points_list[1])
    plt.plot(x, env.demand_points[2].surplus_inventory, label = demand_points_list[2])
    plt.plot(x, env.demand_points[3].surplus_inventory, label = demand_points_list[3])
    plt.plot(x, env.demand_points[4].surplus_inventory, label = demand_points_list[4])
    plt.xlabel('Days')
    plt.ylabel('Surplus Inventory')
    plt.legend()
    plt.title('Surplus Inventory by days')
