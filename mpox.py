# monkeypox simulation
# objects

# WARNING! Check the mutation function below! (from line 269)


import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from graphviz import Digraph

# #Parameters

# #size of the whole population
# size = 10000
# #ratio of the general population
# rGP = .95
# #parameter of the exponentially distributed latent period
# delta0 = .1
# #parameter of the exponentially distributed infectious period
# gamma0 = .05
# #number of sexual contacts in the general population
# cs_gp = .125
# #number of sexual contacts in the core group
# cs_cg = 1.375
# #number of aerial contacts
# ca = 10
# #probability of sexual transmission
# ps = .1
# #probability of aerial transmission
# pa = .00125
# #inherited ratio for the mutations 
# ir = .5
# #standard deviation of the mutations 
# # sigma = 0.0004
# sigma = 0.0004
# #number of initial infectious individuals
# i0 = 10
# #critical outbreak level (percentage)
# crit_outbreak_level = 0.01
# #critical emergency risk (probability)
# crit_emer_risk = 0.05
# #time horizon of the simulation (in days)  /if needed/
# Time = 200


class Individual:
    def __init__(self, exp_time, inf_time, source_group, id, id_from, chain_number, node_type):
        self.exp_time = exp_time
        self.inf_time = inf_time
        self.source_group = source_group
        self.id = id
        self.id_from = id_from
        self.chain_number = chain_number
        self.node_type = node_type

class General(Individual):
    def __init__(self, exp_time, inf_time, rGP, cs, ca, ps, pa, source_group, id, id_from, chain_number, node_type):
        super().__init__(exp_time, inf_time, source_group, id, id_from, chain_number, node_type)
        self.rGP = rGP
        self.cs = cs
        self.ca = ca
        self.ps = ps
        self.pa = pa
        self.calculate_infection_parameters()

    def calculate_infection_parameters(self):
        self.beta0_gp = self.cs * self.ps + self.ca * self.pa * self.rGP
        self.beta0_cg = self.ca * self.pa * (1 - self.rGP)

    @property
    def R0(self):
        return self.beta0_gp / self.inf_time

class Core(Individual):
    def __init__(self, exp_time, inf_time, rGP, cs, ca, ps, pa, source_group, id, id_from, chain_number, node_type):
        super().__init__(exp_time, inf_time, source_group, id, id_from, chain_number, node_type)
        self.rGP = rGP
        self.cs = cs
        self.ca = ca
        self.ps = ps
        self.pa = pa
        self.calculate_infection_parameters()

    def calculate_infection_parameters(self):
        self.beta0_gp = self.ca * self.pa * self.rGP
        self.beta0_cg = self.cs * self.ps + self.ca * self.pa * (1 - self.rGP)

    @property
    def R0(self):
        return self.beta0_cg / self.inf_time

from typing import List

class Group:
    """
    Represents a group of individuals with exposed, infectious, and recovered subgroups.
    """
    def __init__(self, size: int, delta0: float, gamma0: float, rGP: float, cs: float, ca: int, ps: float, pa: float):
        self.size = size
        self.delta0 = delta0
        self.gamma0 = gamma0
        self.rGP = rGP
        self.susceptible = size
        self.recovered = []
        self.exposed = []
        self.infectious = []

        self.cs = cs
        self.ca = ca
        self.ps = ps
        self.pa = pa

        self.cg_source = 0
        self.gp_source = 0

    def __repr__(self):
        return f"Number of exposed: {len(self.exposed)} Number of infectious: {len(self.infectious)} Number of recovered: {len(self.recovered)} Percentage of infection by cg: {self.cg_infected_prop} "

    @property
    def SEIR(self) -> List[int]:
        return [self.susceptible, len(self.exposed), len(self.infectious), len(self.recovered)]

    @property
    def exposed_parameters(self) -> List[float]:
        return [ind.exp_time for ind in self.exposed]

    @property
    def infectious_gp_parameters(self) -> List[float]:
        return [ind.beta0_gp for ind in self.infectious]

    @property
    def infectious_cg_parameters(self) -> List[float]:
        return [ind.beta0_cg for ind in self.infectious]

    @property
    def recovery_parameters(self) -> List[float]:
        return [ind.inf_time for ind in self.infectious]

    @property
    def cg_infected_prop(self):
        if self.susceptible < self.size:
          return round(self.cg_source / (self.size - self.susceptible), 2)
        else:
          return "NA"

    @property
    def gp_infected_prop(self):
        if self.susceptible < self.size:
          return round(self.gp_source / (self.size - self.susceptible), 2)
        else:
          return "NA"

    def infect(self, exposed_ind: Individual):
        """
        An exposed individual becomes infectious.
        """
        self.exposed.remove(exposed_ind)
        self.infectious.append(exposed_ind)

    def recover(self, infectious_ind: Individual):
        """
        An infectious individual recovers.
        """
        self.infectious.remove(infectious_ind)
        self.recovered.append(infectious_ind)

    @property
    def avg_R0(self) -> float:
        if len(self.infectious) == 0:
            return -float("inf")
        else:
            return np.mean([ind.R0 for ind in self.infectious])

    @property
    def max_R0(self) -> float:
        if len(self.infectious) == 0:
            return -float("inf")
        else:
            return max([ind.R0 for ind in self.infectious])
        
    @property
    def quartile_R0(self):
       if len(self.infectious) == 0:
          return [-np.Inf, np.Inf]
       else:
        return(np.quantile([ind.R0 for ind in self.infectious], [0.25, 0.75]))


class GP(Group):
    """
    Represents the general population group.
    """
    def __init__(self, size: int, delta0: float, gamma0: float, rGP: float, cs: float, ca: int, ps: float, pa: float):
        super().__init__(size, delta0, gamma0, rGP, cs, ca, ps, pa)

    def expose(self, exp_time, inf_time, pa, source_group, id, id_from, chain_number, node_type):
      if self.susceptible > 0:
          self.susceptible -= 1
          new_exposed = General(exp_time, inf_time, self.rGP, self.cs, self.ca, self.ps, pa, source_group, id, id_from, chain_number, node_type)
          if source_group == 0:
            self.gp_source += 1
          elif source_group == 1:
            self.cg_source += 1
          self.exposed.append(new_exposed)

class CG(Group):
    """
    Represents the core group.
    """
    def __init__(self, size: int, delta0: float, gamma0: float, rGP: float, cs: float, ca: int, ps: float, pa: float):
        super().__init__(size, delta0, gamma0, rGP, cs, ca, ps, pa)

    def expose(self, exp_time, inf_time, pa, source_group, id, id_from, chain_number, node_type):
      if self.susceptible > 0:
          self.susceptible -= 1
          new_exposed = Core(exp_time, inf_time, self.rGP, self.cs, self.ca, self.ps, pa, source_group, id, id_from, chain_number, node_type)
          if source_group == 0:
            self.gp_source += 1
          elif source_group == 1:
            self.cg_source += 1
          self.exposed.append(new_exposed)

class Population:
  def __init__(self, size, rGP, delta0, gamma0, cs_gp, cs_cg, ca, ps, pa, ir, sigma, n, i0, crit_outbreak_level, crit_emer_risk):
    self.rGP = rGP
    self.delta0 = delta0
    self.gamma0 = gamma0
    self.cs_gp = cs_gp
    self.cs_cg = cs_cg
    self.ca = ca
    self.ps = ps
    self.pa = pa
    self.ir = ir
    self.sigma = sigma
    self.n = n
    self.i0 = i0
    self.crit_outbreak_level = crit_outbreak_level
    self.crit_emer_risk = crit_emer_risk
    self.resTime = np.Inf

    #we initialize an empty population with 1 GP and 1 CG
    self.groups = [GP(size*rGP, delta0, gamma0, rGP, cs_gp, ca, ps, pa)] + [CG(size*(1-rGP), delta0, gamma0, rGP, cs_cg, ca, ps, pa)]
    self.S = size
    self.time = 0

    self.T = [self.time]

    self.SS = [[group.SEIR[0]] for group in self.groups]
    self.EE = [[group.SEIR[1]] for group in self.groups]
    self.II = [[group.SEIR[2]] for group in self.groups]
    self.RR = [[group.SEIR[3]] for group in self.groups]
    self.R0 = [[group.avg_R0] for group in self.groups]
    self.R0max = [[group.max_R0] for group in self.groups]
    self.R0quartile = [[group.quartile_R0] for group in self.groups]

  def __repr__(self):
    string = "The time is: " + str(round(self.time,2))
    string = string + "\nGP:\t" + repr(self.groups[0])
    string = string + "\nCG:\t" + repr(self.groups[1])
    return string

  #calculates the initial size of the population
  @property
  def size(self):
    return sum([group.size for group in self.groups])

  # stochastic mutation
  #the mutation function which gives back the initial value with ir probability and mutates it otherwise
  #using a symmetrically truncated normal distribution to the interval [max(0,2*x-1), min(1,2*x)],
  #where mean(=x) is the initial value and the standad deviation is sigma.
  #Namely, if x <= 0.5, then the truncation interval is [0,2*x], and
  # if x > 0.5, then [2*x-1,1]
  def mutation(self, x):
    if np.random.uniform(0,1) < self.ir:
      return x
    else:
      lower, upper = max(0,2*x-1), min(1,2*x)
      mu = x
      X = stats.truncnorm((lower - mu) / self.sigma, (upper - mu) / self.sigma, loc=mu, scale=self.sigma)
      return(X.rvs(1)[0])
    
  # deterministic mutation
  # def mutation(self, x):
  #   # the value of pa such that R0 = 1:
  #   pn = (self.gamma0-self.cs_gp*self.ps)/(self.ca*self.rGP)
  #   # the multiplicative factor for the n-step mutation
  #   q = pow(pn/self.pa,1/self.n)
  #   return(min(1,q*x))


  #next action
  #gives us the weighted parameters such that we have a list for each group with 4 lists inside:
  #list of exposed_time parameters, weighted infectious parameters for gp and cg and lastly recovery
  #to choose group that acts
  @property
  def weighted_parameters(self):
    return [[
        self.groups[0].exposed_parameters,
        [gp*self.groups[0].susceptible/self.groups[0].size for gp in self.groups[0].infectious_gp_parameters],
        [cg*self.groups[1].susceptible/self.groups[1].size for cg in self.groups[0].infectious_cg_parameters],
        self.groups[0].recovery_parameters
      ]]+[[
        self.groups[1].exposed_parameters,
        [gp*self.groups[0].susceptible/self.groups[0].size for gp in self.groups[1].infectious_gp_parameters],
        [cg*self.groups[1].susceptible/self.groups[1].size for cg in self.groups[1].infectious_cg_parameters],
        self.groups[1].recovery_parameters
      ]]
  #here we have the sum of 4 types of parameters for each group
  #to choose which action
  @property
  def action_agg_param(self):
    return [[sum(v) for v in self.weighted_parameters[group]] for group in range(len(self.groups))]

  #one aggregated parameter to choose group
  @property
  def agg_param(self):
    return [sum(self.action_agg_param[group]) for group in range(len(self.groups))]

  #increases the system's time
  def time_inc(self, inc):
    self.time += inc

  #increases the system's time with the amount of time until the next action
  @property
  def action_time_inc(self):
    self.time_inc(random.expovariate(sum(self.agg_param)))

  #which group initiates the new action
  @property
  def which_group(self):
    return random.choices(range(2),self.agg_param)[0]

  #what type of action the group initiates
  def what_type_of_action(self, groupnum):
    return random.choices(range(4),self.action_agg_param[groupnum])[0]

  #define the exact action that takes place
  def which_action(self, groupnum, actionnum):
    return random.choices(range(len(self.weighted_parameters[groupnum][actionnum])),self.weighted_parameters[groupnum][actionnum])[0]

  #true if no actively exposed or infectious individual is in the population and false otherwise
  @property
  def extinct(self):
    #print(summa)
    summa = sum([len(group.exposed)+len(group.infectious) for group in self.groups])
    if  summa > 0:
      return False
    else:
      return True

  #true if the average of the R0s in GP is higher than 1
  @property
  def evolution(self):
    if self.groups[0].avg_R0 > 1:
      return True
    else:
      return False
  
  #true if the fration of infectious individuals in GP is greater than the critical outbreak level (it's not used)
  @property
  def outbreak(self):
    if self.groups[0].SEIR[2] > self.size * self.crit_outbreak_level:
      return True
    else:
      return False

  #the main step function that determines what happens
  def step(self):
    #the time jumps to the next action's time
    self.action_time_inc
    #which group initiates the action
    action_group = self.which_group
    #what is the type of the action
    action_type = self.what_type_of_action(action_group)
    #which individual makes the action
    action_number = self.which_action(action_group, action_type)
    #number of total infections
    newid = sum([len(group.exposed)+len(group.infectious)+len(group.recovered) for group in self.groups]) + 1

    #if an exposed individual becomes infectious
    if action_type == 0:
      self.groups[action_group].infect(self.groups[action_group].exposed[action_number])
    #if an infectious individual exposes a gp individual
    elif action_type == 1:
      infector = self.groups[action_group].infectious[action_number]
      infector.node_type = "inner"
      self.groups[0].expose(self.delta0, self.gamma0, self.mutation(infector.pa), action_group, newid, infector.id, infector.chain_number + 1, "leaf")
    #if an infectious individual exposes a cg individual
    elif action_type == 2:
      infector = self.groups[action_group].infectious[action_number]
      infector.node_type = "inner"
      self.groups[1].expose(self.delta0, self.gamma0, self.mutation(infector.pa), action_group, newid, infector.id, infector.chain_number + 1, "leaf")
    #if someone recovers
    else:
      recovered = self.groups[action_group].infectious[action_number]
      self.groups[action_group].recover(recovered)
    self.T.append(self.time)

    i = 0
    for group in self.groups:
      self.SS[i].append(group.SEIR[0])
      self.EE[i].append(group.SEIR[1])
      self.II[i].append(group.SEIR[2])
      self.RR[i].append(group.SEIR[3])
      self.R0[i].append(group.avg_R0)
      self.R0max[i].append(group.max_R0)
      self.R0quartile[i].append(group.quartile_R0)
      i += 1

  def transmission_chain(self):
    #chain lengths
    self.chains = []
    # all the GP individuals who connected with the virus
    self.affectedGP = self.groups[0].exposed + self.groups[0].infectious + self.groups[0].recovered
    # all the CG individuals who connected with the virus
    self.affectedCG = self.groups[1].exposed + self.groups[1].infectious + self.groups[1].recovered

    for ind in self.affectedGP + self.affectedCG:
       if ind.node_type == "leaf" or ind.node_type == "single":
          self.chains.append(ind.chain_number)
    return self.chains

       
