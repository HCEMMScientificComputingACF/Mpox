# monkeypox simulation
# create and plot a trajectory
#   with SEIR dynamics and R0 values and
#   with transmission chain graph

# WARNING! Check the mutation function in mpox.py!


from mpox import * # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats

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
# cs_cg = 1.375
cs_cg = 1.375
#number of aerial contacts
ca = 10
#probability of sexual transmission
ps = .1
#probability of aerial transmission
pa = .00125

#inherited ratio for the mutations /if needed, check the mutation function/
ir = .5
#standard deviation of the mutations /if needed, check the mutation function/
sigma = 0.0006
#number of mutation steps /if needed, check the mutation function/
n = 18
#number of initial infectious individuals
i0 = 10
#critical outbreak level (percentage) /if needed, not used here/
crit_outbreak_level = 0.01
#critical emergency risk (probability) /if needed, not used here/
crit_emer_risk = 0.05
#time horizon of the simulation (in days)  /if needed, not used here/
Time = 200

#seeds
seed = 1
random.seed(seed)
np.random.seed(seed=seed)

# which restriction scenario:
# 1 - Scenario I. (baseline, not any restriction)
# 2 - Scenario II (sexual restriction)
# 3 - Scenario III (non-sexual restriction)
scenario = 1

# target population
# a - CG
# b - CG and GP
# c - GP
targetPop = 'a'

# intervention level:
int_level = 0.005
# restriction level:
res_level = 0.5

def makePop():
  # Initialize the population
  population = Population(size, rGP, delta0, gamma0, cs_gp, cs_cg, ca, ps, pa, ir, sigma, n, i0, crit_outbreak_level, crit_emer_risk)

  # Start with i0 many individuals in the core group
  default_gr = population.groups[1]
  # and expose these
  for k in range(i0):
    default_gr.expose(default_gr.delta0, default_gr.gamma0, default_gr.pa, -1, k+1, 0, 0, "single")
  while not population.extinct and not population.evolution:
    if targetPop == 'a':
      targetInf = population.groups[1].SEIR[2]
      targetSize = size * (1-rGP)
    elif targetPop =='b':
      targetInf = population.groups[0].SEIR[2]+population.groups[1].SEIR[2]
      targetSize = size
    elif targetPop =='c':
      targetInf = population.groups[0].SEIR[2]
      targetSize = size * rGP
    if scenario == 2:
      if targetInf >= int_level * targetSize and population.time < population.resTime:
        #set the new value for the new individuals:
        population.groups[1].cs = cs_cg * res_level
        #modify the value for the old individuals:
        for ind in population.groups[1].exposed + population.groups[1].infectious:
          ind.cs = cs_cg * res_level
          ind.calculate_infection_parameters()
        population.resTime = population.time
    elif scenario == 3:
      if targetInf >= int_level * targetSize and population.time < population.resTime:
        #set the new value for the new individuals:
        population.groups[0].ca = ca * res_level
        population.groups[1].ca = ca * res_level
        #modify the value for the old individuals:
        for ind in population.groups[0].exposed + population.groups[0].infectious + population.groups[1].exposed + population.groups[1].infectious:
          ind.ca = ca * res_level
          ind.calculate_infection_parameters()
        population.resTime = population.time 
    population.step()
    print(population)   
  print(population)
  return population

def graph(pop):
    #init of the transmission graph
    tree = Digraph(comment='Tree', format='png')
    tree.attr(rankdir='TB')

    # all the GP individuals who connected with the virus
    affectedGP = pop.groups[0].exposed + pop.groups[0].infectious + pop.groups[0].recovered
    # all the CG individuals who connected with the virus
    affectedCG = pop.groups[1].exposed + pop.groups[1].infectious + pop.groups[1].recovered

    #create the nodes
    for ind in affectedGP:
        #check that the individual has R0 value more than 1 or not:
        if ind.R0 > 1:
            tree.node(str(ind.id), str(ind.chain_number), shape='box', fillcolor='orange', style='filled')
        else:
            tree.node(str(ind.id), str(ind.chain_number), shape='box', fillcolor='lightblue', style='filled')
    for ind in affectedCG:
        #check that the individual in the next exposition to a gp individual induces a R0 value more than 1 or not:
        if (pop.cs_gp*pop.ps+pop.ca*ind.pa)/pop.gamma0 > 1:
            tree.node(str(ind.id), str(ind.chain_number), shape='circle', fillcolor='red', style='filled')
        else:
            tree.node(str(ind.id), str(ind.chain_number), shape='circle', fillcolor='lightblue', style='filled')
    #create the edges
    for ind in affectedGP + affectedCG:
        if ind.id_from != 0:
            tree.edge(str(ind.id_from), str(ind.id))

    #export the graph
    tree.render('transmission_tree', format='png', engine='neato', cleanup=True) # neato: circular layout

# Run 1 population
population = makePop()

# export the transimssion graph
graph(population)

# chain lengths
chains = population.transmission_chain()
print("The length of the chains: ", chains)
print("Mean of the chain lengths: ", np.round(np.mean(chains),2))

# Extract data for plotting
time = population.T
ss_gp = population.SS[0]
ee_gp = population.EE[0]
ii_gp = population.II[0]
rr_gp = population.RR[0]

ss_cg = population.SS[1]
ee_cg = population.EE[1]
ii_cg = population.II[1]
rr_cg = population.RR[1]

# Create subplots
plt.figure(figsize=(12, 6))

# Subplot for General Population (GP)
plt.subplot(2, 2, 1)
#plt.plot(time, ss_gp, label='Susceptible', color='b')
plt.step(time, ee_gp, label='Exposed', color='y')
plt.step(time, ii_gp, label='Infectious', color='r')
plt.step(time, rr_gp, label='Recovered', color='g')
# plt.step(time, [size*crit_outbreak_level for _ in range(len(time))], linestyle='dashed')
plt.axvline(x = population.resTime, linestyle="dashed")
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Dynamics for General Population (GP)')
plt.legend()
plt.xlim(0,time[-1])

# Subplot for Core Group (CG)
plt.subplot(2, 2, 2)
plt.step(time, ss_cg, label='Susceptible', color='b')
plt.step(time, ee_cg, label='Exposed', color='y')
plt.step(time, ii_cg, label='Infectious', color='r')
plt.step(time, rr_cg, label='Recovered', color='g')
plt.axvline(x = population.resTime, linestyle="dashed")
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Dynamics for Core Group (CG)')
plt.legend()
plt.xlim(0,time[-1])

# Subplot for GP: R0 (average; quartiles)
plt.subplot(2, 2, 3)
plt.step(time, population.R0[0], label='average')
# plt.step(population.T[1:], population.R0max[0], label='max')
# plt.step(population.T[1:], population.R0quartile[0], label = 'quartiles')
plt.fill_between(time, np.array(population.R0quartile[0]).T[0], np.array(population.R0quartile[0]).T[1], color='b', alpha=.3, step = 'pre', label = 'quartiles')
# plt.step(population.T[1:], [crit_emer_risk for _ in range(len(population.T[1:]))], linestyle='dashed')
plt.title(r"$R_0$ in GP")
plt.legend()
plt.xlim(0,time[-1])



# Subplot for CG: R0 (average, max)
plt.subplot(2, 2, 4)
plt.step(time, population.R0[1], label='average')
# plt.step(population.T[1:], population.R0max[1], label='max')
plt.title(r"$R_0$ in CG")
plt.legend()
plt.xlim(0,time[-1])

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
