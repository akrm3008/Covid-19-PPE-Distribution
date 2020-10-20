import numpy as np
import pandas as pd 

# Let us create an environment for our problem
# Lets create class for generating regions/demand points
class demand_point:

  def __init__(self, index):
    self.index = index                      # index
    self.time_horizon = None                # time horizon
    self.day = -1                           # day of hospitalisation
    self.inventory = np.array([])           # inventory of PPEs 
    self.PPE_usage = np.array([])           # array containing daily usage of PPEs
    self.surplus_inventory = np.array([])   # array containing surplus daily inventory  
    self.supply = np.array([])              # supply of PPE through regular supply chains 
    self.hospitalisations = np.array([])    # hospitalisation data 
    self.projection = np.array([])          # projection for hospitalisation data 
    self.discharges = np.array([])          # discharges             
    self.deaths = np.array([])              # number of dates per number of admitted people 
    self.active_cases = np.array([])        # active cases 
    self.burn_rates = np.array([])          # number of PPEs used per day per active_case 
    self.vehicles = np.array([])            # vehicles availabe at the demand point
    self.vehicle_capacity = 1000            # capacity of one vehicle
    self.transfers_in = np.array([])        # Inventory transfers from other demand points
    self.transfers_out = np.array([])       # Inventory transfers to other demand points 
    self.vehicles_in = np.array([])         # Vehiceles in from other demand points carrying inventory
    self.vehicles_out = np.array([])        # Vehiceles going to other demand points carrying inventory

# A method for expanding data structures for a new time horizon
  def intialise_new_horizon(self, number_of_days):
    self.time_horizon = number_of_days
    self.day = self.day + 1 
    self.inventory = np.concatenate((self.inventory, np.zeros(self.time_horizon)), axis = 0)
    self.PPE_usage = np.concatenate((self.PPE_usage, np.zeros(self.time_horizon)), axis = 0)
    self.surplus_inventory = np.concatenate((self.surplus_inventory, np.zeros(self.time_horizon)), axis = 0)
    self.active_cases = np.concatenate((self.active_cases, np.zeros(self.time_horizon)), axis = 0)
    self.vehicles = np.concatenate((self.vehicles, np.zeros(self.time_horizon)), axis = 0)
    self.transfers_in = np.concatenate((self.transfers_in, np.zeros(self.time_horizon)), axis = 0)
    self.transfers_out = np.concatenate((self.transfers_out, np.zeros(self.time_horizon)), axis = 0)
    self.vehicles_in = np.concatenate((self.vehicles_in, np.zeros(self.time_horizon)), axis = 0)
    self.vehicles_out = np.concatenate((self.vehicles_out, np.zeros(self.time_horizon)), axis = 0)

# A method for inputing/simulating initial inventories for a region/hospital
  def initial_inv(self, mean = None, sd = None, init_inventory = None):  
    if mean is not None and sd is not None: 
       self.inventory[self.day] = np.random.normal(mean, sd)
    elif init_inventory is not None:
      self.inventory[self.day] = init_inventory
    else:
       self.inventory[self.day] = self.inventory[self.day - 1]

# A method to initialise number of vehicles at the demand point
  def initial_vehicles(self, init_vehicles = 0):
    if init_vehicles != 0:
      self.vehicles[self.day] = init_vehicles
    else: 
      self.vehicles[self.day] = self.vehicles[self.day - 1]

# A method for inputing/simulating hospitalisation following normal/ homogenous Poisson/non-homogenous Poisson
  def get_hospitalisations(self, h_mean = None, h_sd = None, lam = None, lams = None, hospitalisations = None, hospitalisation_trend = None):
    if h_mean is not None and h_sd is not None: 
      self.hospitalisations = np.concatenate((self.hospitalisations, np.random.normal(h_mean, h_sd, self.time_horizon)), axis = 0)
    elif lam is not None: 
      simulation = np.random.poisson(lam, self.time_horizon)
      if hospitalisation_trend == 'up':
        simulation.sort()
      elif hospitalisation_trend == 'down':
        simulation[::-1].sort()
      self.hospitalisations = np.concatenate((self.hospitalisations, simulation), axis = 0)
    elif lams is not None:
      if hospitalisation_trend == 'up':
        lams.sort()
      elif hospitalisation_trend == 'down':
        lams[::-1].sort()
      self.hospitalisations = np.concatenate((self.hospitalisations, np.zeros(self.time_horizon)), axis = 0)
      for i in range(self.day, self.day + len(self.hospitalisations)): 
        self.hospitalisations[i]  = np.random.poisson(lams[i])
    else:
      self.hospitalisations = np.concatenate((self.hospitalisations,hospitalisations), axis = 0)

# A method for inputing simulating deaths and discharges (independent of hospitalisation numbers)
  def get_discharge_deaths(self, deaths_mean, deaths_sd, dis_mean, dis_sd, deaths = None, dis = None):
    if deaths is None: 
      self.deaths = np.concatenate((self.deaths, np.random.normal(deaths_mean, deaths_sd, self.time_horizon)), axis = 0)
    else: 
      self.deaths = np.concatenate((self.deaths, deaths), axis = 0 )
    if dis is None: 
      self.discharges = np.concatenate((self.discharges, np.random.normal(dis_mean, dis_sd, self.time_horizon)), axis = 0)
    else: 
      self.discharges = np.concatenate((self.discharges, dis), axis = 0)

# A method for inputing simulating deaths and discharges (based on hospitalisation numbers)
  def get_discharge_deaths2(self, death_rate, avg_LOS_death, avg_LOS_discharge, deaths = None, dis = None):  # LOS is length of stay 
    self.deaths = np.concatenate((self.deaths, np.zeros(self.time_horizon)), axis = 0)
    self.discharges = np.concatenate((self.discharges, np.zeros(self.time_horizon)), axis = 0)
    for i in range(self.day, len(self.hospitalisations)):
      for j in range(int(self.hospitalisations[i])):
        if np.random.uniform(0,1) > death_rate: 
          LOS = round(np.random.exponential(avg_LOS_discharge))
          if i + LOS < self.time_horizon:
            self.discharges[i + LOS] = self.discharges[i + LOS] + 1 
        else: 
          LOS = round(np.random.exponential(avg_LOS_death))
          if i + LOS < self.time_horizon:
            self.deaths[i + LOS] = self.deaths[i + LOS] + 1 
          
# A method for inputing/simulating projection 
  def projections_simulator(self, mean, sd, projections = None):
    if projections is None: 
      self.projections = np.concatenate((self.projections, np.random.normal(mean, sd, self.time_horizon)), axis = 0)
    else: 
      self.projections = np.concatenate((self.projections, projections), axis = 0)

# A method for calculating active_cases
  def get_active_cases(self):
    if self.day == 0: 
      self.active_cases[self.day] = self.hospitalisations[self.day]
    else: 
      self.active_cases[self.day] = self.active_cases[self.day - 1] + self.hospitalisations[self.day] - self.deaths[self.day - 1] - \
                                    self.discharges[self.day - 1]
    
# A method for inputing/simulating burn_rate
  def get_burn_rates(self, burn_rates_mean, burn_rates_sd, burn_rates = None):
    if burn_rates is None:
      self.burn_rates = np.concatenate((self.burn_rates, np.random.normal(burn_rates_mean, burn_rates_sd, self.time_horizon)), axis = 0) 
    else: 
      self.burn_rates = np.concatenate(self.burn_rates, burn_rates) 

# A method for inputing/simulating supply from the established supply chain
# We assume that supply amount is going to stay constant and can't keep up to the rising demand. It will be equal to the amounts ordered regularly?
# The arguement supply is to input the supply vector for the given time period 
# The arguement average_interarrival_time is to simulate an exponential arrival of supplySince the establihed supply chain is irregular and disturbed, we could simulate it as exponential 
# The arguement uniform start_index is to determine the starting point of a uniform arrival and uniform interarrival times is the uniform length between arrivals
  def get_supply(self, supply_amount, average_interarrival_time = None, uniform_start_index = None, uniform_interarrival_time = None, supply = None):
      if supply is not None:
        self.supply = np.concatenate((self.supply, supply), axis = 0)
      elif uniform_start_index is not None:
        self.supply = np.concatenate((self.supply, np.zeros(self.time_horizon)), axis = 0)
        x = self.day + uniform_start_index
        while x <= len(self.supply) - 1:
          self.supply[x] = supply_amount
          x = x + uniform_interarrival_time
      elif average_interarrival_time is not None:
        self.supply = np.concatenate((self.supply, np.zeros(self.time_horizon)), axis = 0)
        x = self.day
        x = x + round(np.random.exponential(average_interarrival_time))
        while x <= len(self.supply) - 1:
          self.supply[x] = supply_amount
          x = x + round(np.random.exponential(average_interarrival_time))
    

# A method for sending out/ bringing in vehicles with PPE to/from other demand points
  def get_vehicles_transfers(self, vehicles_transfers_matrix = None,  vehicles_in = 0, vehicles_out = 0): 
    self.vehicles[self.day] = self.vehicles[self.day - 1] + vehicles_in - vehicles_out
    if vehicles_transfers_matrix is not None: 
      self.vehicles_out[self.day] = sum([x for x in vehicles_transfers_matrix[self.index] if x >0 ])
      self.vehicles_in[self.day] = sum([x for x in vehicles_transfers_matrix[:,self.index] if x > 0 ])
      self.vehicles[self.day] = self.vehicles[self.day - 1] + self.vehicles_in[self.day]  - self.vehicles_out[self.day] 


# A method for sending out/ bringing in inventory of PPE to/from other demand points
  def get_inventory_transfers(self, inventory_transfers_matrix = None, transfers_in = 0, transfers_out = 0):
    self.transfers_in[self.day] = transfers_in 
    self.transfers_out[self.day] = transfers_out
    if inventory_transfers_matrix is not None: 
      self.transfers_out[self.day] = sum([x for x in inventory_transfers_matrix[self.index] if x > 0])
      self.transfers_in[self.day] = sum([x for x in inventory_transfers_matrix[:,self.index]if x > 0])


# A method to determine if the inventory trasfers made out of demand points are feasible or not 
  def is_transfer_feasible(self):
    if self.transfers_out[self.day] > self.vehicles_out[self.day]*self.vehicle_capacity :
      return False
    elif self.transfers_out[self.day] > self.inventory[self.day] - self.PPE_usage[self.day] - self.transfers_out[self.day] + \
                                        self.transfers_in[self.day] + self.supply[self.day] :
      return False
    elif self.transfers_out[self.day] == 0:
      return True
    else:
      return True

# A method to for imputing PPE usage or calculating it using burn_rate and active cases
  def get_PPE_usage(self, PPE_usage = None): 
    if PPE_usage is None:
      self.PPE_usage[self.day]  = self.burn_rates[self.day]*self.active_cases[self.day] 
    else: 
      self.PPE_usage  = PPE_usage

# A method for inputing current inventory or calculating current inventory using current usages, trasnfers and supplies
  def get_curr_inventory(self, curr_inventory = None): 
    if curr_inventory is None and self.day != 0 :
      self.inventory[self.day]  = self.inventory[self.day - 1] - self.PPE_usage[self.day - 1] - self.transfers_out[self.day - 1] + \
                                self.transfers_in[self.day - 1] + self.supply[self.day - 1]
    elif curr_inventory is not None: 
      self.inventory = curr_inventory

# A method for inputing or calculating surplus inventory on a particular day
  def get_surplus_inventory(self, surplus_inventory = None):
    if surplus_inventory is None:
      self.surplus_inventory[self.day]  = self.inventory[self.day] - self.PPE_usage[self.day] 
    else: 
      self.surplus_inventory = surplus_inventory

# A method to return the state of the demand point
  def get_state(self):
      return np.array([self.index, self.inventory[self.day], self.active_cases[self.day], self.burn_rates[self.day], self.vehicles[self.day], self.vehicle_capacity])

# A method to simulate one step of whole process (i.e for one day) and output state
  def step(self, vehicles_transfers_matrix = None, inventory_transfers_matrix = None): 
    done = False
    if vehicles_transfers_matrix is not None and inventory_transfers_matrix is not None: 
      self.get_vehicles_transfers(vehicles_transfers_matrix)
      self.get_inventory_transfers(inventory_transfers_matrix)
    is_transfer_feasible = self.is_transfer_feasible()
    self.get_active_cases()
    self.get_PPE_usage() 
    self.get_curr_inventory()
    self.get_surplus_inventory()
    surplus_inventory = self.surplus_inventory[self.day]
    state = self.get_state()
    if self.day == len(self.inventory) - 1:
      done = True
    else:
      self.day = self.day + 1
    return is_transfer_feasible, surplus_inventory, state, done