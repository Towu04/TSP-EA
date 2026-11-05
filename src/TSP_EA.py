import src.Reporter as Reporter
import numpy as np
import numba as nb
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

# --- OBJECTIVE FUNCTION ---
def compute_cost(distanceMatrix, succ):
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

			# ----- SELECTION -----
			# TODO

			# ----- RECOMBINATION -----
			# TODO

			# ----- MUTATION -----
			# TODO

			# ----- ELMINATION -----
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