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