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
def initialize_population(
    distance_matrix: FloatMatrix, pop_size: int
) -> list[Individual]:
    """Initialize the population with valid, locally optimized tours."""
    n: int = distance_matrix.shape[0]
    population: list[Individual] = []

    while len(population) < pop_size:
        tour: IntArray = np.random.permutation(n).astype(np.int32)
        
        # 1. Repair missing edges first
        if np.isinf(compute_cost_tour(distance_matrix, tour)):
            tour = repair_tour(distance_matrix, tour)
            
        # 2. Enrich with Or-opt to improve starting fitness
        tour = or_opt_asymmetric(distance_matrix, tour)
        
        # 3. Recompute final cost (and retry in the rare case a dead-end remained)
        cost: float = compute_cost_tour(distance_matrix, tour)
        if np.isinf(cost):
            raise Exception

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
        v: int = tour[(i + 1) % n]
        distance: float = distance_matrix[u, v]
        if np.isinf(distance):
            raise Exception("this tour is invalid, something went wrong in the initialization or recombination")
        cost += distance
    return cost


@nb.njit
def repair_tour(distance_matrix: FloatMatrix, tour: IntArray) -> IntArray:
    """Repairs np.inf edges via forward swaps and targeted node relocations."""
    n: int = tour.shape[0]

    # Phase 1: Forward greedy swaps for indices 0 .. n-2
    for i in range(n - 1):
        u: int = tour[i]
        v: int = tour[i + 1]
        if np.isinf(distance_matrix[u, v]):
            # Pick a random starting offset among the remaining unvisited cities (j > i)
            rem: int = n - (i + 1)
            start_offset: int = np.random.randint(0, rem)
            for offset in range(rem):
                j: int = (i + 1) + ((start_offset + offset) % rem)
                cand: int = tour[j]
                if not np.isinf(distance_matrix[u, cand]):
                    tour[i + 1], tour[j] = tour[j], tour[i + 1]
                    break

    # Phase 2: Clean up the closing edge (or any stuck edge) via swap/relocation
    for _pass in range(10):
        has_inf: bool = False
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            if np.isinf(distance_matrix[u, v]):
                has_inf = True
                start_offset = np.random.randint(0, n)
                for offset in range(n):
                    j = (start_offset + offset) % n
                    if j == i or j == (i + 1) % n:
                        continue
                    prev_j: int = (j - 1) % n
                    next_j: int = (j + 1) % n
                    next_i: int = (i + 2) % n
                    cand = tour[j]

                    # Check if swapping v (at i+1) and cand (at j) creates 4 valid edges
                    if (
                        not np.isinf(distance_matrix[u, cand])
                        and not np.isinf(distance_matrix[cand, tour[next_i]])
                        and not np.isinf(distance_matrix[tour[prev_j], v])
                        and not np.isinf(distance_matrix[v, tour[next_j]])
                    ):
                        tour[(i + 1) % n], tour[j] = tour[j], tour[(i + 1) % n]
                        break
        if not has_inf:
            break

    return tour


def compute_pop_fitness(population: list[Individual]) -> tuple[float, Individual]:
    """Return the (average_cost, best_individual) of a non-empty population."""
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