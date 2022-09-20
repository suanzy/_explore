#%%
""" 
Getting started in 30s
- Import libraries
- Define your fitness function
- Define a problem dictionary
- Training and get the results
"""
from mealpy.evolutionary_based import GA
import numpy as np

def fitness_func(solution):
    return np.sum(solution**2)

problem_dict = {
    "fit_func": fitness_func,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "minmax": "min",
    "log_to": "file",
    "log_file": "result.log"
}

ga_model = GA.BaseGA(problem_dict, epoch=100, pop_size=50, pc=0.85, pm=0.1)
best_position, best_fitness_value = ga_model.solve()

print(best_position)
print(best_fitness_value)
# %%
""" Fitness Function Preparation """
from opfunu.type_based.uni_modal import Functions           # or
# from opfunu.cec.cec2014 import Functions                     # or
# from opfunu.dimension_based.benchmarknd import Functions

# Then you need to create an object of Function to get the functions
type_based = Functions()
F1 = type_based._sum_squres__
# F2 = type_based.__dixon_price__

#%%
import numpy as np

## This is normal fitness function
def fitness_normal(solution=None):
        return np.sqrt(solution**2)         # Single value


## This is how you design multi-objective function
#### Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
def fitness_multi(solution):
    def booth(x, y):
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2
    def bukin(x, y):
        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
    def matyas(x, y):
        return 0.26 * (x**2 + y**2) - 0.48 * x * y
    return [booth(solution[0], solution[1]), bukin(solution[0], solution[1]), matyas(solution[0], solution[1])]


## This is how you design Constrained Benchmark Function (G01)
#### Link: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119136507.app2
def fitness_constrained(solution):
    def g1(x):
        return 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
    def g2(x):
        return 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10
    def g3(x):
        return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
    def g4(x):
        return -8 * x[0] + x[9]
    def g5(x):
        return -8 * x[1] + x[10]
    def g6(x):
        return -8 * x[2] + x[11]
    def g7(x):
        return -2 * x[3] - x[4] + x[9]
    def g8(x):
        return -2 * x[5] - x[6] + x[10]
    def g9(x):
        return -2 * x[7] - x[8] + x[11]

    def violate(value):
        return 0 if value <= 0 else value

    fx = 5 * np.sum(solution[:4]) - 5 * np.sum(solution[:4] ** 2) - np.sum(solution[4:13])

    ## Increase the punishment for g1 and g4 to boost the algorithm (You can choice any constraint instead of g1 and g4)
    fx += violate(g1(solution)) ** 2 + violate(g2(solution)) + violate(g3(solution)) + \
        2 * violate(g4(solution)) + violate(g5(solution)) + violate(g6(solution)) + \
        violate(g7(solution)) + violate(g8(solution)) + violate(g9(solution))
    return fx

#%%
""" Problem Preparation 
- fit_func: Your fitness function
- lb: Lower bound of variables, it should be list of values
- ub: Upper bound of variables, it should be list of values
- minmax: The problem you are trying to solve is minimum or maximum, value can be “min” or “max”
- obj_weights: list weights for all your objectives (Optional, default = [1, 1, …1])
"""

## Design a problem dictionary for normal function
problem_normal = {
    "fit_func": fitness_normal,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "minmax": "min",
}

## Design a problem dictionary for multiple objective functions above
problem_multi = {
    "fit_func": fitness_multi,
    "lb": [-10, -10],
    "ub": [10, 10],
    "minmax": "min",
    "obj_weights": [0.4, 0.1, 0.5]               # Define it or default value will be [1, 1, 1]
}

## Design a problem dictionary for constrained objective function above
problem_constrained = {
  "fit_func": fitness_constrained,
  "lb": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "ub": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1],
  "minmax": "min",
}
#%%
""" Need to import the algorithm that will be used
1. process: Using multi-cores to update fitness for whole population (Parallel: no effect on updating process)
2. thread: Using multi-threads to update fitness for whole population (Parallel: no effect on updating process)
3. swarm: Updating fitness after the whole population move (Sequential: no effect on updating process)
4. single: Updating fitness after each agent move (Sequential: effect on updating process)

"""
from mealpy.bio_based import SMA
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO

sma_model = SMA.BaseSMA(problem_normal, epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness_value = sma_model.solve()   # default is: single

sma_model = SMA.BaseSMA(problem_normal, epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness_value = sma_model.solve(mode="single")

sma_model = SMA.BaseSMA(problem_normal, epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness_value = sma_model.solve(mode="swarm")

ga_model = GA.BaseGA(problem_multi, epoch=1000, pop_size=100, pc=0.9, pm=0.05)
best_position, best_fitness_value = ga_model.solve(mode="thread")

pso_model = PSO.BasePSO(problem_constrained, epoch=500, pop_size=80, c1=2.0, c2=1.8, w_min=0.3, w_max=0.8)
best_position, best_fitness_value = pso_model.solve(mode="process")


#%%
""" Advances -- Lower/Upper Bound """
# When you have different lower bound and upper bound for each parameters
problem_dict1 = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
}

#%%
# When you have same lower bound and upper bound for each variable, then you can use:

## 2.1 number: then you need to specify your problem size / number of dimensions (n_dims)
problem_dict2 = {
   "fit_func": F5,
   "lb": -10,
   "ub": 30,
   "minmax": "min",
   "n_dims": 30,  # Remember the keyword "n_dims"
}

## 2.2 array: Then there are 2 ways
problem_dict3 = {
   "fit_func": F5,
   "lb": [-5],
   "ub": [10],
   "minmax": "min",
   "n_dims": 30,  # Remember the keyword "n_dims"
}

## or
n_dims = 100
problem_dict4 = {
   "fit_func": F5,
   "lb": [-5] * n_dims,
   "ub": [10] * n_dims,
   "minmax": "min",
}


#%%
""" Advances -- Stopping Condition (Termination) """
# 1. FE (Number of Function Evaluation)
term_dict1 = {
   "mode": "FE",
   "quantity": 100000    # 100000 number of function evaluation
}

#%%
#2. MG (Maximum Generations / Epochs): This is default in all algorithms
term_dict2 = {  # When creating this object, it will override the default epoch you define in your model
   "mode": "MG",
   "quantity": 1000  # 1000 epochs
}

#%%
# 3. ES (Early Stopping): Same idea in training neural network (If the global best solution not better an epsilon after K epoch then stop the program

term_dict3 = {
   "mode": "ES",
   "quantity": 30  # after 30 epochs, if the global best doesn't improve then we stop the program
}

#%%
# 4. TB (Time Bound): You just want your algorithm run in K seconds. Especially when comparing different algorithms
term_dict4 = {
   "mode": "TB",
   "quantity": 60  # 60 seconds = 1 minute to run this algorithm only
}

#%%
# After import and create a termination object, you need to pass it to your optimizer as a additional parameter with the keyword “termination”
model3 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03, termination=term_dict4)
model3.solve()

#%%
""" Advances -- Problem Preparation """
""" 1. WARNING: The memory issues related to mealpy:
By default, the history of population is saved. This can cause the memory issues if your problem is too big. You can set “save_population” = False to avoid this. However, you won't be able to draw the trajectory chart of agents.
"""
problem_dict1 = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
   "log_to": "console",
   "save_population": False,              # Default = True
}

#%%
""" 2. Logging results of training process: 
3 options:
- Log to console (default): problem_dict1
- Log to file: problem_dict2
- Don’t show log: problem_dict3
"""
problem_dict1 = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
   "log_to": "console",              # Default
}

problem_dict2 = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
   "log_to": "file",
   "log_file": "result.log",         # Default value = "mealpy.log"
}

problem_dict3 = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
   "log_to": None,
}


#%%
""" 3. Set up necessary functions for discrete problem:
Let’s say we want to solve Travelling Salesman Problem (TSP), we need to design at least a function that generate solution, a function that bring back the solution to the boundary, and finally the fitness function"""

import numpy as np


def generate_position(lb=None, ub=None):
    ## For Travelling Salesman Problem, the solution should be a permutation
    ## Lowerbound: [0, 0,...]
    ## Upperbound: [N_cities - 1.11, ....]
    return np.random.permutation(len(lb))

def amend_position(solution, lb=None, ub=None):
    # print(f"Raw: {solution}")
    ## Bring them back to boundary
    solution = np.clip(solution, lb, ub)

    solution_set = set(list(range(0, len(solution))))
    solution_done = np.array([-1, ] * len(solution))
    solution_int = solution.astype(int)
    city_unique, city_counts = np.unique(solution_int, return_counts=True)

    for idx, city in enumerate(solution_int):
        if solution_done[idx] != -1:
            continue
        if city in city_unique:
            solution_done[idx] = city
            city_unique = np.where(city_unique == city, -1, city_unique)
        else:
            list_cities_left = list(solution_set - set(city_unique) - set(solution_done))
            solution_done[idx] = list_cities_left[0]
    # print(f"Final solution: {solution_done}")
    return solution_done

def fitness_function(solution):
    ## Objective for this problem is the sum of distance between all cities that salesman has passed
    ## This can be change depend on your requirements
    city_coord = CITY_POSITIONS[solution]
    line_x = city_coord[:, 0]
    line_y = city_coord[:, 1]
    total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))
    return total_distance

problem = {
    "fit_func": fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",
    "log_to": "console",
    "generate_position": generate_position,
    "amend_position": amend_position,
}


#%%
""" Advances -- Model Definition """
# 1. Name the optimizer model and name the fitness function:
from mealpy.swarm_based import PSO

problem = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
}

model = PSO.BasePSO(problem, epoch=10, pop_size=50, name="Normal PSO", fit_name="Benchmark Function 5th")

print(model.name)
print(model.fit_name)

#%%
# 2. Set up Stopping Condition for an optimizer:
term_dict = {
   "mode": "TB",
   "quantity": 60  # 60 seconds = 1 minute to run this algorithm only
}

model = PSO.BasePSO(problem, epoch=100, pop_size=50, termination=term_dict)
model.solve()


#%%
# Hint Validation for setting up the hyper-parameters:

# If you don’t know how to set up hyper-parameters and valid range for it. Try to set different type for that hyper-parameter.

model = PSO.BasePSO(problem, epoch="hello", pop_size="world")
model.solve()

# $ 2022/03/22 08:59:16 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'epoch' is an integer and value should be in range: [1, 100000].

model = PSO.BasePSO(problem, epoch=10, pop_size="world")
model.solve()

# $ 2022/03/22 09:01:51 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'pop_size' is an integer and value should be in range: [10, 10000].

#%%
""" Advances --  More on Fitnesss Function"""
# Usually, when defining a fitness function we only need 1 parameter which is the solution.
def fitness_function(solution):
    fitness = np.sum(solution**2)
    return fitness

#%%
# version 2.4.2, you can define your data (whatever it is) as an input parameter to fitness function
from mealpy.swarm_based import PSO

def fitness_function(solution, data):
    dataset = data['dataset']
    additional_infor = data['additional-information']
    network = NET(dataset, additional_infor)
    fitness = network.loss
    return fitness

DATA = {
    "dataset": dataset,
    "additional-information": temp,
}

problem = {
   "fit_func": F5,
   "lb": [-3, -5, 1, -10, ],
   "ub": [5, 10, 100, 30, ],
   "minmax": "min",
   "data": DATA,     # Remember this keyword 'data'
}

model = PSO.BasePSO(problem, epoch=10, pop_size=50)
model.solve()

#%%
""" Advances -- Starting Positions """
# Not recommended to use this utility. But in case you need this:

from mealpy.human_based import TLO
import numpy as np

def frequency_modulated(pos):
        # range: [-6.4, 6.35], f(X*) = 0, phi = 2pi / 100
        phi = 2 * np.pi / 100
        result = 0
        for t in range(0, 101):
                y_t = pos[0] * np.sin(pos[3] * t * phi + pos[1]*np.sin(pos[4] * t * phi + pos[2] * np.sin(pos[5] * t * phi)))
                y_t0 = 1.0 * np.sin(5.0 * t * phi - 1.5 * np.sin(4.8 * t * phi + 2.0 * np.sin(4.9 * t * phi)))
                result += (y_t - y_t0)**2
        return result

fm_problem = {
        "fit_func": frequency_modulated,
        "lb": [-6.4, ] * 6,
        "ub": [6.35, ] * 6,
        "minmax": "min",
        "log_to": "console",
        "save_population": False,
}
term_dict1 = {
   "mode": "FE",
   "quantity": 5000    # 100000 number of function evaluation
}

## This is an example I use to create starting positions
## Write your own function, remember the starting positions has to be: list of N vectors or 2D matrix of position vectors
def create_starting_positions(n_dims=None, pop_size=None, num=1):
        return np.ones((pop_size, n_dims)) * num + np.random.uniform(-1, 1)

## Define the model
model = TLO.BaseTLO(fm_problem, epoch=100, pop_size=50, termination=term_dict1)

## Input your starting positions here
list_pos = create_starting_positions(6, 50, 2)
best_position, best_fitness = model.solve(None, starting_positions=list_pos)        ## Remember the keyword: starting_positions
print(f"Best solution: {model.solution}, Best fitness: {best_fitness}")

## Training with other starting positions
list_pos2 = create_starting_positions(6, 50, -1)
best_position, best_fitness = model.solve(None, starting_positions=list_pos2)
print(f"Best solution: {model.solution}, Best fitness: {best_fitness}")



#%%
""" Advances -- Agent's History

You can access to the history of agent/population in model.history object with variables:
- list_global_best: List of global best SOLUTION found so far in all previous generations
- list_current_best: List of current best SOLUTION in each previous generations
- list_epoch_time: List of runtime for each generation
- list_global_best_fit: List of global best FITNESS found so far in all previous generations
- list_current_best_fit: List of current best FITNESS in each previous generations
- list_diversity: List of DIVERSITY of swarm in all generations
- list_exploitation: List of EXPLOITATION percentages for all generations
- list_exploration: List of EXPLORATION percentages for all generations
- list_population: List of POPULATION in each generations #!cause the error related to ‘memory’ when saving model. Better to set parameter ‘save_population’ to False in the input problem dictionary to not using it.
"""
import numpy as np
from mealpy.swarm_based.PSO import BasePSO

def fitness_function(solution):
    return np.sum(solution**2)

problem_dict = {
    "fit_func": fitness_function,
    "lb": [-10, -15, -4, -2, -8],
    "ub": [10, 15, 12, 8, 20],
    "minmax": "min",
    "verbose": True,
    "save_population": False        # Then you can't draw the trajectory chart
}
model = BasePSO(problem_dict, epoch=1000, pop_size=50)

print(model.history.list_global_best)
print(model.history.list_current_best)
print(model.history.list_epoch_time)
print(model.history.list_global_best_fit)
print(model.history.list_current_best_fit)
print(model.history.list_diversity)
print(model.history.list_exploitation)
print(model.history.list_exploration)
print(model.history.list_population)

## Remember if you set "save_population" to False, then there is no variable: list_population


#%%
""" Advances -- Import All Models """
from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, TPO, VCS, WHO
from mealpy.evolutionary_based import CRO, DE, EP, ES, FPA, GA, MA
from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, ICA, LCO, QSA, SARO, SSDO, TLO
from mealpy.math_based import AOA, CGO, GBO, HC, SCA, PSS
from mealpy.music_based import HS
from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, SA, TWO, WDO
from mealpy.probabilistic_based import CEM
from mealpy.system_based import AEO, GCO, WCA
from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BeesA, BES, BFO, BSA, COA, CSA, CSO, DO, EHO, FA, FFA, FOA, GOA, GWO, HGS
from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, WOA

import numpy as np


def fitness_function(solution):
    return np.sum(solution ** 2)


problem = {
    "fit_func": fitness_function,
    "lb": [-3],
    "ub": [5],
    "n_dims": 30,
    "save_population": False,
    "log_to": None,
    "log_file": "results.log"
}

if __name__ == "__main__":
    ## Run the algorithm
    model = BBO.BaseBBO(problem_dict1, epoch=31, pop_size=10, p_m=0.01, elites=3, termination=term_dict, name="HGS", fit_name="F0")
    model = EOA.BaseEOA(problem, epoch=10, pop_size=50, p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=0.9, gamma=0.9)
    model = IWO.OriginalIWO(problem, epoch=10, pop_size=50, seeds=(2, 10), exponent=2, sigmas=(0.1, 0.001))
    model = SBO.BaseSBO(problem, epoch=10, pop_size=50, alpha=0.94, p_m=0.05, psw=0.02)
    model = SBO.OriginalSBO(problem, epoch=10, pop_size=50, alpha=0.94, p_m=0.05, psw=0.02)
    model = SMA.OriginalSMA(problem, epoch=10, pop_size=50, p_t=0.1, termination=term_dict)
    model = SMA.BaseSMA(problem, epoch=10, pop_size=50, p_t=0.1, termination=term_dict)
    model = VCS.OriginalVCS(problem, epoch=10, pop_size=50, lamda=0.99, xichma=0.2)
    model = VCS.BaseVCS(problem, epoch=10, pop_size=50, lamda=0.99, xichma=2.2)
    model = WHO.BaseWHO(problem, epoch=10, pop_size=50, n_s=3, n_e=3, eta=0.15, p_hi=0.9, local_move=(0.9, 0.3), global_move=(0.2, 0.8), delta=(2.0, 2.0))
    model = CRO.BaseCRO(problem_dict1, epoch=20, pop_size=50, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, G=(0.05, 0.21), n_trials=5)
    model = CRO.OCRO(problem_dict1, epoch=20, pop_size=50, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, G=(0.05, 0.21), n_trials=5, restart_count=10)
    model = DE.BaseDE(problem_dict1, epoch=20, pop_size=50, wf=0.1, cr=0.9, strategy=5)
    model = DE.JADE(problem_dict1, epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1)
    model = DE.SADE(problem_dict1, epoch=20, pop_size=50)
    model = DE.SHADE(problem_dict1, epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5)
    model = DE.L_SHADE(problem_dict1, epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5)
    model = DE.SAP_DE(problem_dict1, epoch=20, pop_size=50, branch="ABS")
    model = EP.BaseEP(problem_dict1, epoch=20, pop_size=50, bout_size=0.2)
    model = EP.LevyEP(problem_dict1, epoch=20, pop_size=50, bout_size=0.2)
    model = ES.BaseES(problem_dict1, epoch=20, pop_size=50, lamda=0.7)
    model = ES.LevyES(problem_dict1, epoch=20, pop_size=50, lamda=0.7)
    model = FPA.BaseFPA(problem_dict1, epoch=20, pop_size=50, p_s=0.8, levy_multiplier=0.2)
    model = GA.BaseGA(problem_dict1, epoch=10, pop_size=50, pc=0.95, pm=0.1, mutation_multipoints=True, mutation="flip")
    model = MA.BaseMA(problem_dict1, epoch=10, pop_size=50, pc=0.85, pm=0.1, p_local=0.5, max_local_gens=25, bits_per_param=4)
    model = BRO.OriginalBRO(problem_dict1, epoch=100, pop_size=50, threshold=1)
    model = BRO.BaseBRO(problem_dict1, epoch=10, pop_size=50, threshold=1)
    model = BSO.ImprovedBSO(problem_dict1, epoch=10, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5)
    model = BSO.BaseBSO(problem_dict1, epoch=10, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5, slope=30)
    model = CA.OriginalCA(problem_dict1, epoch=10, pop_size=50, accepted_rate=0.15)
    model = CHIO.OriginalCHIO(problem_dict1, epoch=10, pop_size=50, brr=0.15, max_age=2)
    model = CHIO.BaseCHIO(problem_dict1, epoch=10, pop_size=50, brr=0.15, max_age=1)
    model = FBIO.OriginalFBIO(problem_dict1, epoch=11, pop_size=50)
    model = FBIO.BaseFBIO(problem_dict1, epoch=10, pop_size=50)
    model = GSKA.OriginalGSKA(problem_dict1, epoch=10, pop_size=50, pb=0.1, kf=0.5, kr=0.9, kg=2)
    model = GSKA.BaseGSKA(problem_dict1, epoch=10, pop_size=50, pb=0.1, kr=0.7)
    model = ICA.BaseICA(problem_dict1, epoch=10, pop_size=50, empire_count=5, assimilation_coeff=1.5,
                        revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1, zeta=0.1)
    model = LCO.OriginalLCO(problem_dict1, epoch=10, pop_size=50, r1=2.35)
    model = LCO.BaseLCO(problem_dict1, epoch=10, pop_size=50, r1=2.35)
    model = LCO.ImprovedLCO(problem_dict1, epoch=10, pop_size=50)
    model = QSA.BaseQSA(problem_dict1, epoch=10, pop_size=50)
    model = QSA.OriginalQSA(problem_dict1, epoch=10, pop_size=50)
    model = QSA.OppoQSA(problem_dict1, epoch=10, pop_size=50)
    model = QSA.LevyQSA(problem_dict1, epoch=10, pop_size=50)
    model = QSA.ImprovedQSA(problem_dict1, epoch=10, pop_size=50)
    model = SARO.OriginalSARO(problem_dict1, epoch=10, pop_size=50, se=0.5, mu=5)
    model = SARO.BaseSARO(problem_dict1, epoch=10, pop_size=50, se=0.5, mu=25)
    model = SSDO.BaseSSDO(problem_dict1, epoch=10, pop_size=50)
    model = TLO.BaseTLO(problem_dict1, epoch=10, pop_size=50)
    model = TLO.OriginalTLO(problem_dict1, epoch=10, pop_size=50)
    model = TLO.ITLO(problem_dict1, epoch=10, pop_size=50, n_teachers=5)
    model = AOA.OriginalAOA(problem_dict1, epoch=30, pop_size=50, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9)
    model = CGO.OriginalCGO(problem_dict1, epoch=10, pop_size=50)
    model = GBO.OriginalGBO(problem_dict1, epoch=10, pop_size=50, pr=0.5, beta_minmax=(0.2,1.2))
    model = HC.OriginalHC(problem_dict1, epoch=10, pop_size=50, neighbour_size=20)
    model = HC.BaseHC(problem_dict1, epoch=10, pop_size=50, neighbour_size=20)
    model = PSS.OriginalPSS(problem_dict1, epoch=10, pop_size=50, acceptance_rate=0.9, sampling_method="MC")
    model = SCA.BaseSCA(problem_dict1, epoch=10, pop_size=50)
    model = SCA.OriginalSCA(problem_dict1, epoch=10, pop_size=50)
    model = HS.BaseHS(problem_dict1, epoch=10, pop_size=50)
    model = HS.OriginalHS(problem_dict1, epoch=10, pop_size=50)
    model = ArchOA.OriginalArchOA(problem_dict1, epoch=10, pop_size=50, c1=2, c2=6, c3=2, c4=0.5, acc_max=0.9, acc_min=0.1)
    model = ArchOA.OriginalArchOA(problem_dict1, epoch=10, pop_size=50, alpha=50, beta=0.2)
    model = EFO.OriginalEFO(problem_dict1, epoch=10, pop_size=50, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
    model = EFO.BaseEFO(problem_dict1, epoch=10, pop_size=50, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
    model = EO.BaseEO(problem_dict1, epoch=10, pop_size=50)
    model = EO.ModifiedEO(problem_dict1, epoch=10, pop_size=50)
    model = EO.AdaptiveEO(problem_dict1, epoch=10, pop_size=50)
    model = HGSO.BaseHGSO(problem_dict1, epoch=10, pop_size=50, n_clusters=3)
    model = MVO.BaseMVO(problem_dict1, epoch=10, pop_size=50, wep_min=0.2, wep_max=1.0)
    model = MVO.OriginalMVO(problem_dict1, epoch=10, pop_size=50, wep_min=0.2, wep_max=1.0)
    model = NRO.BaseNRO(problem_dict1, epoch=10, pop_size=50)
    model = SA.BaseSA(problem_dict1, epoch=10, pop_size=50, max_sub_iter=5, t0=1000, t1=1, move_count=5, mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99)
    model = TWO.BaseTWO(problem_dict1, epoch=10, pop_size=50)
    model = TWO.OppoTWO(problem_dict1, epoch=10, pop_size=50)
    model = TWO.LevyTWO(problem_dict1, epoch=100, pop_size=50)
    model = TWO.EnhancedTWO(problem_dict1, epoch=10, pop_size=50)
    model = WDO.BaseWDO(problem, epoch=100, pop_size=50, RT=3, g_c=0.2, alp=0.4, c_e=0.4, max_v=0.3)
    model = CEM.BaseCEM(problem_dict1, epoch=10, pop_size=50, n_best=40, alpha=0.5)
    model = AEO.OriginalAEO(problem_dict1, epoch=10, pop_size=50)
    model = AEO.AdaptiveAEO(problem_dict1, epoch=10, pop_size=50)
    model = AEO.ModifiedAEO(problem_dict1, epoch=10, pop_size=50)
    model = AEO.EnhancedAEO(problem_dict1, epoch=10, pop_size=50)
    model = AEO.IAEO(problem_dict1, epoch=10, pop_size=50)
    model = GCO.BaseGCO(problem_dict1, epoch=10, pop_size=50, cr=0.7, wf=1.25)
    model = GCO.OriginalGCO(problem_dict1, epoch=10, pop_size=50, cr=0.7, wf=1.25)
    model = WCA.BaseWCA(problem_dict1, epoch=10, pop_size=50, nsr=4, wc=2.0, dmax=1e-6)
    model = ABC.BaseABC(problem_dict1, epoch=10, pop_size=50, couple_bees=(16, 4), patch_variables=(5.0, 0.985), sites=(3, 1))
    model = ACOR.BaseACOR(problem, epoch=10, pop_size=50, sample_count=25, intent_factor=0.5, zeta=1.0)
    model = ALO.OriginalALO(problem, epoch=10, pop_size=50)
    model = ALO.BaseALO(problem, epoch=10, pop_size=50)
    model = AO.OriginalAO(problem, epoch=10, pop_size=50)
    model = BA.BaseBA(problem, epoch=10, pop_size=50, loudness=(1.0, 2.0), pulse_rate=(0.15, 0.85), pulse_frequency=(0, 10))
    model = BA.OriginalBA(problem, epoch=10, pop_size=50, loudness=0.8, pulse_rate=0.95, pulse_frequency=(0, 10))
    model = BA.ModifiedBA(problem, epoch=10, pop_size=50, pulse_rate=0.95, pulse_frequency=(0, 10))
    model = BeesA.BaseBeesA(problem_dict1, epoch=10, pop_size=50, site_ratio=(0.5, 0.4), site_bee_ratio=(0.1, 2.0), dance_factor=(0.1, 0.99))
    model = BeesA.BaseBeesA(problem_dict1, epoch=10, pop_size=50, recruited_bee_ratio=0.1, dance_factor=(0.1, 0.99))
    model = BES.BaseBES(problem_dict1, epoch=10, pop_size=50, a_factor=10, R_factor=1.5, alpha=2.0, c1=2.0, c2=2.0)
    model = BFO.OriginalBFO(problem_dict1, epoch=10, pop_size=50, Ci=0.01, Ped=0.25, Nc=5, Ns=4, attract_repels=(0.1, 0.2, 0.1, 10))
    model = BFO.ABFO(problem_dict1, epoch=10, pop_size=50, Ci=(0.1, 0.001), Ped=0.01, Ns=4, N_minmax=(2, 40))
    model = BSA.BaseBSA(problem_dict1, epoch=10, pop_size=50, ff=10, pff=0.8, c_couples=(1.5, 1.5), a_couples=(1.0, 1.0), fl=0.5)
    model = BSA.BaseBSA(problem_dict1, epoch=10, pop_size=50, ff=10, pff=0.8, c_couples=(1.5, 1.5), a_couples=(1.0, 1.0), fl=0.5)
    model = COA.BaseCOA(problem_dict1, epoch=10, pop_size=50, n_coyotes=5)
    model = CSA.BaseCSA(problem_dict1, epoch=10, pop_size=50, p_a=0.7)
    model = CSO.BaseCSO(problem_dict1, epoch=10, pop_size=50, mixture_ratio=0.15, smp=5,
                        spc=False, cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1)
    model = DO.BaseDO(problem_dict1, epoch=10, pop_size=50)
    model = EHO.BaseEHO(problem_dict1, epoch=10, pop_size=50, alpha=0.5, beta=0.5, n_clans=5)
    model = FA.BaseFA(problem_dict1, epoch=10, pop_size=50, max_sparks=10, p_a=0.04, p_b=0.8, max_ea=30, m_sparks=5)
    model = FFA.BaseFFA(problem_dict1, epoch=10, pop_size=50, gamma=0.001, beta_base=2, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2)
    model = FOA.OriginalFOA(problem_dict1, epoch=10, pop_size=50)
    model = FOA.BaseFOA(problem_dict1, epoch=10, pop_size=50)
    model = FOA.WhaleFOA(problem_dict1, epoch=10, pop_size=50)
    model = GOA.BaseGOA(problem_dict1, epoch=10, pop_size=50, c_minmax=(0.00004, 1))
    model = GWO.BaseGWO(problem_dict1, epoch=10, pop_size=50)
    model = GWO.RW_GWO(problem_dict1, epoch=10, pop_size=50)
    model = HGS.OriginalHGS(problem_dict1, epoch=10, pop_size=50, PUP=0.08, LH=10000)
    model = HHO.BaseHHO(problem_dict1, epoch=10, pop_size=50)
    model = JA.BaseJA(problem_dict1, epoch=10, pop_size=50)
    model = JA.OriginalJA(problem_dict1, epoch=10, pop_size=50)
    model = JA.LevyJA(problem_dict1, epoch=10, pop_size=50)
    model = MFO.OriginalMFO(problem_dict1, epoch=10, pop_size=50)
    model = MFO.BaseMFO(problem_dict1, epoch=10, pop_size=50)
    model = MRFO.BaseMRFO(problem_dict1, epoch=10, pop_size=50)
    model = MSA.BaseMSA(problem_dict1, epoch=10, pop_size=50, n_best=5, partition=0.5, max_step_size=1.0)
    model = NMRA.BaseNMRA(problem_dict1, epoch=10, pop_size=50)
    model = NMRA.ImprovedNMRA(problem_dict1, epoch=10, pop_size=50, pb=0.75, pm=0.01)
    model = PFA.BasePFA(problem_dict1, epoch=10, pop_size=50)
    model = PSO.BasePSO(problem_dict1, epoch=100, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
    model = PSO.C_PSO(problem_dict1, epoch=10, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
    model = PSO.CL_PSO(problem_dict1, epoch=10, pop_size=50, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7)
    model = PSO.PPSO(problem_dict1, epoch=10, pop_size=50)
    model = PSO.HPSO_TVAC(problem_dict1, epoch=10, pop_size=50, ci=0.5, cf=0.2)
    model = SFO.BaseSFO(problem_dict1, epoch=10, pop_size=50, pp=0.1, AP=4, epxilon=0.0001)
    model = SFO.ImprovedSFO(problem_dict1, epoch=10, pop_size=50, pp=0.1)
    model = SHO.BaseSHO(problem_dict1, epoch=10, pop_size=50, h_factor=1, rand_v=(0, 2), N_tried=30)
    model = SLO.BaseSLO(problem_dict1, epoch=10, pop_size=50)
    model = SLO.ModifiedSLO(problem_dict1, epoch=10, pop_size=50)
    model = SLO.ISLO(problem_dict1, epoch=10, pop_size=50, c1=1.2, c2=1.2)
    model = SRSR.BaseSRSR(problem_dict1, epoch=10, pop_size=50)
    model = SSA.OriginalSSA(problem_dict1, epoch=10, pop_size=50, ST=0.8, PD=0.2, SD=0.1)
    model = SSA.BaseSSA(problem_dict1, epoch=10, pop_size=50, ST=0.8, PD=0.2, SD=0.1)
    model = SSO.BaseSSO(problem, epoch=10, pop_size=50)
    model = SSpiderA.BaseSSpiderA(problem_dict1, epoch=10, pop_size=50, r_a=1, p_c=0.7, p_m=0.1)
    model = SSpiderO.BaseSSpiderO(problem_dict1, epoch=10, pop_size=50, fp=(0.65, 0.9))
    model = WOA.BaseWOA(problem_dict1, epoch=10, pop_size=50)
    model = WOA.HI_WOA(problem_dict1, epoch=10, pop_size=50, feedback_max=5)

    best_position, best_fitness = model.solve()
    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

#%%

#%%



#%%

#%%


