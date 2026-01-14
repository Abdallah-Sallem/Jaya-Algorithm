# File: jaya_algorithm.py
# Author: Abdallah Sallem
# Description: Full Python implementation of the Jaya Algorithm
# for solving optimization problems (minimization).

import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Objective Function (to minimize)
# Example: f(x, y) = x^2 + y^2
# ===============================
def objective_function(solution):
    """
    Replace this function with any optimization function.
    :param solution: numpy array of variables [x1, x2, ..., xn]
    :return: fitness value
    """
    return np.sum(solution**2)  # simple sphere function

# ===============================
# Initialize Population
# ===============================
def initialize_population(pop_size, dim, lower_bound, upper_bound):
    """
    Initialize a population within given bounds.
    :param pop_size: number of solutions
    :param dim: number of dimensions
    :param lower_bound: lower bound for variables (scalar or array)
    :param upper_bound: upper bound for variables (scalar or array)
    :return: numpy array of shape (pop_size, dim)
    """
    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, dim))
    return population

# ===============================
# Evaluate Population Fitness
# ===============================
def evaluate_population(population):
    """
    Evaluate fitness for each solution in the population.
    :param population: numpy array of solutions
    :return: numpy array of fitness values
    """
    fitness = np.array([objective_function(ind) for ind in population])
    return fitness

# ===============================
# Jaya Algorithm Update Rule
# ===============================
def update_population(population, best_solution, worst_solution):
    """
    Update each solution according to the Jaya algorithm formula.
    :param population: current population
    :param best_solution: current best solution
    :param worst_solution: current worst solution
    :return: updated population
    """
    pop_size, dim = population.shape
    new_population = np.copy(population)
    
    for i in range(pop_size):
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        new_population[i] = population[i] + r1*(best_solution - abs(population[i])) - r2*(worst_solution - abs(population[i]))
    
    return new_population

# ===============================
# Main Jaya Algorithm
# ===============================
def jaya_algorithm(pop_size=20, dim=2, lower_bound=-10, upper_bound=10, iterations=100):
    """
    Run the Jaya Algorithm to minimize the objective function.
    :param pop_size: population size
    :param dim: number of variables
    :param lower_bound: lower bound for variables
    :param upper_bound: upper bound for variables
    :param iterations: number of iterations
    :return: best solution found, fitness over iterations
    """
    population = initialize_population(pop_size, dim, lower_bound, upper_bound)
    fitness_history = []

    for it in range(iterations):
        fitness = evaluate_population(population)
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        best_solution = population[best_idx]
        worst_solution = population[worst_idx]
        
        population = update_population(population, best_solution, worst_solution)
        
        best_fitness = evaluate_population(population).min()
        fitness_history.append(best_fitness)

        # Optional: print progress
        print(f"Iteration {it+1}/{iterations}, Best Fitness: {best_fitness:.5f}")

    # Final best solution
    final_fitness = evaluate_population(population)
    best_idx = np.argmin(final_fitness)
    best_solution = population[best_idx]
    
    return best_solution, fitness_history

# ===============================
# Visualization Function
# ===============================
def plot_convergence(fitness_history):
    """
    Plot the convergence of the algorithm.
    """
    plt.figure(figsize=(8,5))
    plt.plot(fitness_history, marker='o', linestyle='-', color='blue')
    plt.title("Jaya Algorithm Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show()

# ===============================
# Example Run
# ===============================
if __name__ == "__main__":
    best_solution, fitness_history = jaya_algorithm(
        pop_size=30,
        dim=2,
        lower_bound=-10,
        upper_bound=10,
        iterations=50
    )
    print("\nBest Solution Found:", best_solution)
    print("Best Fitness:", objective_function(best_solution))
    
    plot_convergence(fitness_history)
