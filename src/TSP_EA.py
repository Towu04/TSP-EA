import src.Reporter as Reporter
import numpy as np
import numba as nb
from typing import List
from dataclasses import dataclass

# --- REPRESENTATION ---
@dataclass
class Individual:
	def __init__(self, pred: np.ndarray, succ: np.ndarray, cost: int):
		self.pred = pred
		self.succ = succ
		self.cost = cost

	def tour(self):
		# Convert the predecessor/successor representation to a tour representation.
		#Time complexity: O(N)
		N = self.pred.shape[0]
		tour = np.zeros(N, dtype=np.int32)
		current_city = 0
		for i in range(N):
			tour[i] = current_city
			current_city = self.succ[current_city]
		return tour

# --- INITIALIZATION ---
def initialize_population(distanceMatrix: np.ndarray, POP_SIZE: int):
	# Initialize the population with random tours.
	# TODO: 2-opt for fitter initial population
	i = 0
	N = distanceMatrix.shape[0]
	population = list()
	while i < POP_SIZE:
		tour = np.random.permutation(N)
		cost = compute_cost_tour(distanceMatrix, tour)
		if cost == -1:
			break
		succ, pred = tour_to_pred_succ(tour)
		population.append(Individual(pred, succ, cost))
		i += 1
	return population

# --- HELPER FUNCTIONS ---

def compute_cost_succ(distanceMatrix: np.ndarray, succ: np.ndarray):
	# Compute the cost of a tour given its successor representation.
	#Time complexity: O(N)
	N = succ.shape[0]
	cost = 0
	current_city = 0
	for _ in range(N):
		next_city = succ[current_city]
		cost += distanceMatrix[current_city, next_city]
		current_city = next_city
	return cost

def compute_cost_tour(distanceMatrix: np.ndarray, tour: np.ndarray):
	# Compute the cost of a tour given its tour representation.
	# returns -1 if missing edge
	N = tour.shape[0]
	cost = 0
	for i in range(N-1):
		distance = distanceMatrix[tour[i], tour[i+1]]
		if distance == np.inf:
			return -1
		else:
			cost += distance
	return cost

def compute_pop_fitness(population: List[Individual]):
	# find the average and best value of a population
	current_best = population[0].cost
	pop_size = 0
	total_cost = 0

	for individual in population:
		if individual.cost < current_best:
			current_best = individual.cost
		total_cost += individual.cost
		pop_size += 1

	average_cost = total_cost / pop_size

	return average_cost, current_best





def tour_to_pred_succ(tour):
	# convert a tour representation to predecessor and successor representations.
	# time complexity: O(N)
	N = tour.shape[0]
	pred = np.zeros(N, dtype=np.int32)
	succ = np.zeros(N, dtype=np.int32)

	for i in range(N):
		pred[tour[i]] = tour[i-1]
		succ[tour[i]] = tour[i+1]
	return succ, pred

class TSP_EA:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# ----- HYPERPARAMETERS -----
		N = distanceMatrix.shape[0]
		POP_SIZE = 100

		# ----- REPRESENTATION -----
		pop_succ = np.zeros((POP_SIZE, N), dtype=np.int32)
		pop_pred = np.zeros((POP_SIZE, N), dtype=np.int32)
		pop_cost = np.zeros(POP_SIZE, dtype=np.int64)

		# ----- INITIALIZATION -----
		# TODO

		# ----- STOPPING CRITERIA -----
		# TODO
		yourConvergenceTestsHere = True

		while( yourConvergenceTestsHere ):
			# Your code here.
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = np.array([0,1,2,3,4,5,6])

			# ----- RECOMBINATION -----
			# TODO


			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			# Leave the next three lines as they are.
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.
		return 0

__build_hash__ = "447be2744c11fd5d50e5995e5c1801e89fee68b5f22965aecff55f0978405c24"
if __name__ == '__main__':
	# Write any testing code here.
	pass