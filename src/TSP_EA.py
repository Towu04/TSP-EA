import src.Reporter as Reporter
import numpy as np
import numba as nb
from numpy.typing import NDArray
from dataclasses import dataclass

# --- TYPE ALIASES ---
IntArray = NDArray[np.int32]
FloatMatrix = NDArray[np.float64]


# --- REPRESENTATION ---
@dataclass
class Individual:
    pred: IntArray
    succ: IntArray
    cost: float

    def tour(self) -> IntArray:
        """Convert the predecessor/successor representation to a tour array.
        Time complexity: O(N)
        """
        n: int = self.pred.shape[0]
        tour: IntArray = np.zeros(n, dtype=np.int32)
        current_city: int = 0
        for i in range(n):
            tour[i] = current_city
            current_city = int(self.succ[current_city])
        return tour


# --- INITIALIZATION ---
def initialize_population(distance_matrix: FloatMatrix, pop_size: int) -> list[Individual]:
    """Initialize the population with valid random tours."""
    # TODO: 2-opt for fitter initial population
    # TODO: better initialization strategy for missing edges, retries will be redundant
    n: int = distance_matrix.shape[0]
    population: list[Individual] = []

    while len(population) < pop_size:
        tour: IntArray = np.random.permutation(n).astype(np.int32)
        cost: float = compute_cost_tour(distance_matrix, tour)
        if np.isinf(cost):
            continue  # Retry if tour hits an invalid (inf) edge
        succ, pred = tour_to_pred_succ(tour)
        population.append(Individual(pred=pred, succ=succ, cost=cost))

    return population


# --- HELPER FUNCTIONS ---
@nb.njit
def compute_cost_succ(distance_matrix: FloatMatrix, succ: IntArray) -> float:
    """Compute the cost of a tour given its successor representation.
    Time complexity: O(N)
    """
    n: int = succ.shape[0]
    cost: float = 0.0
    current_city: int = 0
    for _ in range(n):
        next_city: int = succ[current_city]
        distance: float = distance_matrix[current_city, next_city]
        if np.isinf(distance):
            return np.inf
        cost += distance
        current_city = next_city
    return cost


@nb.njit
def compute_cost_tour(distance_matrix: FloatMatrix, tour: IntArray) -> float:
    """Compute the cost of a tour given its tour permutation representation.
    Returns np.inf if the tour uses a missing/infinite edge.
    """
    n: int = tour.shape[0]
    cost: float = 0.0
    for i in range(n):
        u: int = tour[i]
        v: int = tour[(i + 1) % n]  # Wraps around to close the tour at i == n - 1
        distance: float = distance_matrix[u, v]
        if np.isinf(distance):
            return np.inf
        cost += distance
    return cost


def compute_pop_fitness(population: list[Individual]) -> tuple[float, Individual]:
    """Return the (average_cost, best_individual) of a non-empty population."""
    if not population:
        raise ValueError("Population cannot be empty.")

    best_ind: Individual = min(population, key=lambda ind: ind.cost)
    total_cost: float = sum(ind.cost for ind in population)
    average_cost: float = total_cost / len(population)

    return average_cost, best_ind


@nb.njit
def tour_to_pred_succ(tour: IntArray) -> tuple[IntArray, IntArray]:
    """Convert a 1D tour permutation to (succ, pred) arrays.
    Time complexity: O(N)
    """
    n: int = tour.shape[0]
    pred: IntArray = np.zeros(n, dtype=np.int32)
    succ: IntArray = np.zeros(n, dtype=np.int32)

    for i in range(n):
        city: int = tour[i]
        pred[city] = tour[i - 1]
        succ[city] = tour[(i + 1) % n]

    return succ, pred


class TSP_EA:

    def __init__(self) -> None:
        self.reporter: Reporter.Reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename: str) -> int:
        # Read distance matrix from file as float64 (supports np.inf)
        with open(filename, "r") as file:
            distance_matrix: FloatMatrix = np.loadtxt(
                file, delimiter=",", dtype=np.float64
            )

        # ----- HYPERPARAMETERS -----
        n: int = distance_matrix.shape[0]
        pop_size: int = 100

        # ----- INITIALIZATION -----
        population: list[Individual] = initialize_population(distance_matrix, pop_size)

        # ----- STOPPING CRITERIA -----
        your_convergence_tests_here: bool = True

        while your_convergence_tests_here:
            mean_objective, best_individual = compute_pop_fitness(population)
            best_objective: float = best_individual.cost
            best_solution: IntArray = best_individual.tour()

            # ----- RECOMBINATION -----
            # TODO: EAX crossover + survival selection

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            # Leave the next three lines as they are.
            time_left: float = self.reporter.report(
                mean_objective, best_objective, best_solution
            )
            if time_left < 0:
                break

        return 0


__build_hash__: str = "447be2744c11fd5d50e5995e5c1801e89fee68b5f22965aecff55f0978405c24"
if __name__ == "__main__":
    pass