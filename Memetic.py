import random
from typing import List, Tuple

# Tham số
GENE_LENGTH = 100
POP_SIZE = 300
GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2
LOCAL_SEARCH_RATE = 0.1
TOUR_SIZE = 3

def create_individual(gene_length: int = GENE_LENGTH) -> List[int]:
    return [random.randint(0, 1) for _ in range(gene_length)]

def init_population(pop_size: int = POP_SIZE, gene_length: int = GENE_LENGTH) -> List[List[int]]:
    return [create_individual(gene_length) for _ in range(pop_size)]

def eval_one_max(individual: List[int]) -> int:
    return sum(individual)

def tournament_select(pop: List[List[int]], k: int = TOUR_SIZE) -> List[int]:
    contenders = random.sample(pop, k)
    return max(contenders, key=eval_one_max)

def cx_one_point(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mut_flip_bit(individual: List[int], indpb: float) -> List[int]:
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]
    return individual

# Local Search: 1-bit flip scan, trả về cá thể tốt nhất tìm được
def local_search(individual: List[int]) -> List[int]:
    best = individual.copy()
    best_fit = eval_one_max(best)
    for i in range(len(individual)):
        neighbor = best.copy()
        neighbor[i] = 1 - neighbor[i]
        f = eval_one_max(neighbor)
        if f > best_fit:
            best_fit = f
            best = neighbor
    return best

def main_memetic_ga() -> Tuple[List[int], int]:
    pop = init_population(POP_SIZE, GENE_LENGTH)

    # đánh giá ban đầu và hall-of-fame
    best_ind = max(pop, key=eval_one_max)
    best_fit = eval_one_max(best_ind)

    for gen in range(GENERATIONS):
        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)
            if random.random() < CX_PROB:
                c1, c2 = cx_one_point(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if random.random() < MUT_PROB:
                c1 = mut_flip_bit(c1, indpb=0.05)
            if random.random() < MUT_PROB:
                c2 = mut_flip_bit(c2, indpb=0.05)
            offspring.append(c1)
            if len(offspring) < POP_SIZE:
                offspring.append(c2)

        # Áp dụng local search (memetic) cho khoảng LOCAL_SEARCH_RATE của quần thể
        for i in range(len(offspring)):
            if random.random() < LOCAL_SEARCH_RATE:
                offspring[i] = local_search(offspring[i])

        pop = offspring

        # cập nhật best
        gen_best = max(pop, key=eval_one_max)
        gen_best_fit = eval_one_max(gen_best)
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = gen_best.copy()

    return best_ind, best_fit

if __name__ == "__main__":
    best_individual, best_fitness = main_memetic_ga()
    print(f"Best fitness: {best_fitness}/{GENE_LENGTH}")
    print(f"Best individual (first 100 bits): {best_individual}")