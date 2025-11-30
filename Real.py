import random
from typing import List, Tuple

# Tham số
NUM_VARS = 100
POP_SIZE = 300
GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2
TOUR_SIZE = 3
LOW = 0.0
UP = 1.0
BLEND_ALPHA = 0.1  # cho cxBlend
MUT_INDPROB = 0.05
MUT_RANGE = 0.1    # biên độ thay đổi khi mutate (sẽ giới hạn vào [LOW,UP])

def create_individual(n: int = NUM_VARS) -> List[float]:
    return [random.random() * (UP - LOW) + LOW for _ in range(n)]

def init_population(pop_size: int = POP_SIZE, n: int = NUM_VARS) -> List[List[float]]:
    return [create_individual(n) for _ in range(pop_size)]

def eval_real(individual: List[float]) -> float:
    # Ví dụ: tối đa hóa tổng các biến
    return sum(individual)

def tournament_select(pop: List[List[float]], k: int = TOUR_SIZE) -> List[float]:
    contenders = random.sample(pop, k)
    return max(contenders, key=eval_real)

def cx_blend(parent1: List[float], parent2: List[float], alpha: float = BLEND_ALPHA) -> Tuple[List[float], List[float]]:
    """Blend crossover (cxBlend) cho mỗi gene."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1)):
        d = abs(parent1[i] - parent2[i])
        low = min(parent1[i], parent2[i]) - alpha * d
        up = max(parent1[i], parent2[i]) + alpha * d
        val1 = random.uniform(low, up)
        val2 = random.uniform(low, up)
        # Clip vào [LOW, UP]
        child1[i] = max(LOW, min(UP, val1))
        child2[i] = max(LOW, min(UP, val2))
    return child1, child2

def mutate_real(individual: List[float], indpb: float = MUT_INDPROB, mut_range: float = MUT_RANGE) -> List[float]:
    """Đột biến: với prob indpb cho mỗi gene, thêm nhiễu đồng nhất và clip vào [LOW,UP]."""
    for i in range(len(individual)):
        if random.random() < indpb:
            delta = random.uniform(-mut_range, mut_range)
            individual[i] = max(LOW, min(UP, individual[i] + delta))
    return individual

def main_real_ga() -> Tuple[List[float], float]:
    pop = init_population(POP_SIZE, NUM_VARS)
    best_ind = max(pop, key=eval_real)
    best_fit = eval_real(best_ind)

    for gen in range(GENERATIONS):
        offspring: List[List[float]] = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)
            if random.random() < CX_PROB:
                c1, c2 = cx_blend(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if random.random() < MUT_PROB:
                c1 = mutate_real(c1)
            if random.random() < MUT_PROB:
                c2 = mutate_real(c2)
            offspring.append(c1)
            if len(offspring) < POP_SIZE:
                offspring.append(c2)

        pop = offspring
        gen_best = max(pop, key=eval_real)
        gen_best_fit = eval_real(gen_best)
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = gen_best.copy()

    return best_ind, best_fit

if __name__ == "__main__":
    best_individual, best_fitness = main_real_ga()
    print(f"Best fitness: {best_fitness}/{NUM_VARS * UP}")
    print(f"Best individual (first 10 vars): {best_individual[:10]}")