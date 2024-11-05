# monkeypox simulation
# simulation study for the stochastic mutation case (varying the value of sigma)

# WARNING! Check the mutation function in mpox.py!


import numpy as np
import scipy.stats as stats
import datetime
from joblib import Parallel, delayed
import multiprocessing
from mpox import * # type: ignore
from pandas import DataFrame

#Parameters

#size of the whole population
size = 10000
#ratio of the general population
rGP = .95
#parameter of the exponentially distributed latent period
delta0 = .1
#parameter of the exponentially distributed infectious period
gamma0 = .05
#number of sexual contacts in the general population
cs_gp = .125
#number of sexual contacts in the core group
cs_cg = 1.375
#number of aerial contacts
ca = 10
#probability of sexual transmission
ps = .1
#probability of aerial transmission
pa = .00125
#number of initial infectious individuals
i0 = 10
#critical outbreak level (percentage) /if needed, not used here/
crit_outbreak_level = 0.01
#critical emergency risk (probability) /if needed, not used here/
crit_emer_risk = 0.05
#time horizon of the simulation (in days)  /if needed, not used here/
Time = 200
# n-step mutation level /not used here/
n = 1

#inherited ratio for the mutations 
ir = .5
#number of sigmas
num_sigma = 7
#list of sigmas
# sigmaList = np.linspace(0.0003,0.0006,num_sigma)
sigmaList = [0.0006]
# sigmaList = (np.logspace(0, np.log10(6), num_sigma+1)[1:] - 1) / 10000

#number of runs
num_simulations = 1000

# which restriction scenario:
# 1 - Scenario I. (baseline, not any restriction)
# 2 - Scenario II/b. (sexual restriction)
# 3 - Scenario III/b. (non-sexual restriction)
scenario = 1

# intervention level:
int_level = 0.005
# restriction level:
res_level = 0.5


def makeSim(sigma):
  # Initialize the population
  population = Population(size, rGP, delta0, gamma0, cs_gp, cs_cg, ca, ps, pa, ir, sigma, n, i0, crit_outbreak_level, crit_emer_risk)

  # Start with i0 many individuals in the core group
  default_gr = population.groups[1]
  # and expose these
  for k in range(i0):
    default_gr.expose(default_gr.delta0, default_gr.gamma0, default_gr.pa, -1, k+1, 0, 0, "single")
  
  while not population.extinct and not population.evolution:
    if scenario == 2:
      if population.groups[0].SEIR[2] + population.groups[1].SEIR[2]>= int_level*size and population.time < population.resTime:
        #set the new value for the new individuals:
        population.groups[1].cs = cs_cg * res_level
        #modify the value for the old individuals:
        for ind in population.groups[1].exposed + population.groups[1].infectious:
          ind.cs = cs_cg * res_level
          ind.calculate_infection_parameters()
        population.resTime = population.time
    elif scenario == 3:
      if population.groups[0].SEIR[2] + population.groups[1].SEIR[2] >= int_level*size and population.time < population.resTime:
        #set the new value for the new individuals:
        population.groups[0].ca = ca * res_level
        population.groups[1].ca = ca * res_level
        #modify the value for the old individuals:
        for ind in population.groups[0].exposed + population.groups[0].infectious + population.groups[1].exposed + population.groups[1].infectious:
          ind.ca = ca * res_level
          ind.calculate_infection_parameters()
        population.resTime = population.time 
    population.step()
  print("sigma = ", sigma)
  print(population)
  if population.evolution:
    evol = 1
  else:
    evol = 0
  if population.extinct:
    extinction = 1
  else:
    extinction = 0
  endTime = population.time
  chain = np.mean(population.transmission_chain())
  return [evol, extinction, endTime, population.resTime, chain]


startTime = datetime.datetime.now()
print("List of sigma-s: ", sigmaList)

# Run the simulations parallelly:
results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(makeSim)(sigma) for sigma in sigmaList for _ in range(num_simulations))

# Extract the results:
evolList = []
extinctionList = []
endTimeList = []
resTimeList = []
chainList = []
for i in range(len(sigmaList)):
  tr = np.array(results[(num_simulations * i):(num_simulations * (i+1))]).T
  evolList.append(sum(tr[0]))
  extinctionList.append(sum(tr[1]))
  endTimeList.append(sum(tr[2]))
  resTimeList.append(sum(tr[3]))
  chainList.append(sum(tr[4]))
endTimeList = list(np.round([x/num_simulations for x in endTimeList], 2))
resTimeList = list(np.round([x/num_simulations for x in resTimeList], 2))
chainList = list(np.round([x/num_simulations for x in chainList], 2))



# Print the results to the console:
# Print the results to the console:
print("Run time: ", datetime.datetime.now() - startTime)
print("Evolutions: ", evolList)
print("Extinctions: ", extinctionList)
print("Mean durations: ", endTimeList)
print("Mean restriction times: ", resTimeList)
print("Mean length of the chains: ", chainList)
print("List of sigma-s: ", sigmaList)
print("Number of runs: ", num_simulations)

# Export the result to an excel file:
df = DataFrame({'sigma': sigmaList, 'Evolutions': evolList, 'Extinctions': extinctionList, 'Durations': endTimeList, 'Intervention time': resTimeList, "Chain": chainList})
df.to_excel('r.xlsx', sheet_name='sheet1', index=False)