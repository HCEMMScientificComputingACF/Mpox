# monkeypox simulation
# simulation study for the deterministic mutation case (varying the value of n)

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
#inherited ratio for the mutations 
ir = .5
#number of initial infectious individuals
i0 = 10
#critical outbreak level (percentage) /if needed, not used here/
crit_outbreak_level = 0.01
#critical emergency risk (probability) /if needed, not used here/
crit_emer_risk = 0.05
#time horizon of the simulation (in days)  /if needed, not used here/
Time = 200

#standard deviation of the mutations /if needed, check the mutation function/
sigma = 0.0007

#number of mutation steps /if needed, check the mutation function/
nList = [i+12 for i in range(15)] 

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


def makeSim(n):
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
  print("n = ", n)
  print(population)
  if population.evolution:
    evolution = 1
  else:
    evolution = 0
  if population.extinct:
    extinction = 1
  else:
    extinction = 0
  endTime = population.time
  # mean of the transmission chains:
  chain = np.mean(population.transmission_chain())
  return [evolution, extinction, endTime, population.resTime, chain]


startTime = datetime.datetime.now()
print("List of n-s: ", nList)

# Run the simulations parallelly:
results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(makeSim)(n) for n in nList for _ in range(num_simulations))

# Extract the results:
evolutionList = []
extinctionList = []
endTimeList = []
resTimeList = []
chainList = []
for i in range(len(nList)):
  tr = np.array(results[(num_simulations * i):(num_simulations * (i+1))]).T
  evolutionList.append(sum(tr[0]))
  extinctionList.append(sum(tr[1]))
  endTimeList.append(sum(tr[2]))
  resTimeList.append(sum(tr[3]))
  chainList.append(sum(tr[4]))
endTimeList = list(np.round([x/num_simulations for x in endTimeList], 2))
resTimeList = list(np.round([x/num_simulations for x in resTimeList], 2))
chainList = list(np.round([x/num_simulations for x in chainList], 2))



# Print the results to the console:
print("Run time: ", datetime.datetime.now() - startTime)
print("Evolutions: ", evolutionList)
print("Extinctions: ", extinctionList)
print("Mean durations: ", endTimeList)
print("Mean restriction times: ", resTimeList)
print("Mean length of the chains: ", chainList)
print("List of n-s: ", nList)
print("Number of runs: ", num_simulations)

# Export the result to an excel file:
df = DataFrame({'n': nList, 'Evolutions': evolutionList, 'Extinctions': extinctionList, 'Durations': endTimeList, 'Intervention time': resTimeList, 'Chain length:': chainList})
df.to_excel('r.xlsx', sheet_name='sheet1', index=False)