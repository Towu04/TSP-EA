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
    """Initialize the population with valid, Or-opt refined tours."""
    population: list[Individual] = []

    while len(population) < pop_size:
        succ, pred = generate_and_repair_succ_pred(distance_matrix)

        # Ensure Phase 2 eliminated all inf edges before running local search
        if np.isinf(compute_cost_succ(distance_matrix, succ)):
            continue

        or_opt_1(distance_matrix, succ, pred)
        cost: float = compute_cost_succ(distance_matrix, succ)

        population.append(Individual(pred=pred, succ=succ, cost=cost))

    return population

# --- LOCAL SEARCH OPERATORS ---

@nb.njit
def or_opt_1(
    distance_matrix: FloatMatrix, succ: IntArray, pred: IntArray
) -> None:
    """In-place O(N^2) Or-opt (k=1) using succ/pred pointers for O(1) node relocation."""
    n: int = succ.shape[0]

    for _pass in range(n):
        city_order: IntArray = np.random.permutation(n).astype(np.int32)
        improved_in_pass: bool = False

        for idx in range(n):
            city_A: int = city_order[idx]
            prev_city_A: int = pred[city_A]
            next_city_A: int = succ[city_A]

            # Pick a random target city_X to insert city_A right after (into edge X -> Y)
            city_X: int = np.random.randint(0, n)

            # Skip if edge (X -> Y) is already one of the two edges incident to city_A
            if city_X == prev_city_A or city_X == city_A:
                continue

            city_Y: int = succ[city_X]

            insert_cost: float = (
                distance_matrix[prev_city_A, next_city_A]
                + distance_matrix[city_X, city_A]
                + distance_matrix[city_A, city_Y]
            )

            # Never introduce an inf edge
            if np.isinf(insert_cost):
                continue

            remove_cost: float = (
                distance_matrix[prev_city_A, city_A]
                + distance_matrix[city_A, next_city_A]
                + distance_matrix[city_X, city_Y]
            )

            if insert_cost >= remove_cost:
                continue

            # --- O(1) Pointer Update ---
            # 1. Stitch prev_city_A directly to next_city_A (bypassing city_A)
            succ[prev_city_A] = next_city_A
            pred[next_city_A] = prev_city_A

            # 2. Splice city_A between city_X and city_Y
            succ[city_X] = city_A
            pred[city_A] = city_X

            succ[city_A] = city_Y
            pred[city_Y] = city_A

            improved_in_pass = True

        # Early exit if a full pass found no random improvements
        if not improved_in_pass:
            break

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
def generate_and_repair_succ_pred(
    distance_matrix: FloatMatrix,
) -> tuple[IntArray, IntArray]:
    """Generates a random tour, builds valid forward edges on the permutation,
    converts to (succ, pred), and repairs any remaining inf edges via O(1) pointer moves.
    """
    n: int = distance_matrix.shape[0]
    tour: IntArray = np.random.permutation(n).astype(np.int32)

    # --- PHASE 1: Forward greedy swaps on the 1D array (indices 0 .. n-2) ---
    for i in range(n - 1):
        u: int = tour[i]
        if np.isinf(distance_matrix[u, tour[i + 1]]):
            rem: int = n - (i + 1)
            start_offset: int = np.random.randint(0, rem)
            for offset in range(rem):
                j: int = (i + 1) + ((start_offset + offset) % rem)
                if not np.isinf(distance_matrix[u, tour[j]]):
                    tour[i + 1], tour[j] = tour[j], tour[i + 1]
                    break

    # Convert to (succ, pred) once so Phase 2 and Or-Opt can do O(1) pointer updates
    succ, pred = tour_to_pred_succ(tour)

    # --- PHASE 2: Repair remaining inf edges (e.g. closing edge) on succ/pred ---
    for _pass in range(10):
        has_inf: bool = False
        for u in range(n):
            v: int = succ[u]
            if not np.isinf(distance_matrix[u, v]):
                continue

            has_inf = True
            w: int = succ[v]
            start_city: int = np.random.randint(0, n)
            repaired: bool = False

            # Option A (3-edge constraint): Relocate `v` into edge (x -> y)
            # Requires (u -> w) to be valid, and (x -> v -> y) to be valid.
            if not np.isinf(distance_matrix[u, w]):
                for offset in range(n):
                    x: int = (start_city + offset) % n
                    if x == u or x == v:
                        continue
                    y: int = succ[x]
                    if not np.isinf(distance_matrix[x, v]) and not np.isinf(
                        distance_matrix[v, y]
                    ):
                        # 1. Bypass v: stitch u -> w
                        succ[u] = w
                        pred[w] = u
                        # 2. Insert v between x and y: stitch x -> v -> y
                        succ[x] = v
                        pred[v] = x
                        succ[v] = y
                        pred[y] = v
                        repaired = True
                        break

            # Option B (4-edge constraint fallback): Swap `v` with `cand` in-place
            # Used if (u -> w) is also inf, preventing Option A.
            if not repaired:
                for offset in range(n):
                    cand: int = (start_city + offset) % n
                    # Skip adjacent nodes so the 4 edges don't overlap
                    if cand == u or cand == v or cand == w or succ[cand] == u:
                        continue
                    cand_prev: int = pred[cand]
                    cand_next: int = succ[cand]

                    if (
                        not np.isinf(distance_matrix[u, cand])
                        and not np.isinf(distance_matrix[cand, w])
                        and not np.isinf(distance_matrix[cand_prev, v])
                        and not np.isinf(distance_matrix[v, cand_next])
                    ):
                        # Put cand between u and w
                        succ[u] = cand
                        pred[cand] = u
                        succ[cand] = w
                        pred[w] = cand

                        # Put v between cand_prev and cand_next
                        succ[cand_prev] = v
                        pred[v] = cand_prev
                        succ[v] = cand_next
                        pred[cand_next] = v
                        break

        if not has_inf:
            break

    return succ, pred


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