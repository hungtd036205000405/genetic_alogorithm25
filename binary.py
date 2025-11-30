import random
from typing import List, Tuple

# GA params
GENE_LENGTH = 100
POP_SIZE = 300
GENERATIONS = 50
TOUR_SIZE = 3

# Adaptive control params (có thể chỉnh)
PC_MAX = 0.8
PM_BASE = 0.05
MAX_FIT = GENE_LENGTH  # One-Max tối đa

class Individual:
    def __init__(self, genes: List[int]):
        self.genes = genes
        self.fitness: int = None

    def copy(self):
        ind = Individual(self.genes.copy())
        ind.fitness = self.fitness
        return ind

    def __repr__(self):
        return f"Ind(f={self.fitness}, genes={self.genes[:10]}...)"

def create_individual(length: int = GENE_LENGTH) -> Individual:
    return Individual([random.randint(0, 1) for _ in range(length)])

def init_population(pop_size: int = POP_SIZE, length: int = GENE_LENGTH) -> List[Individual]:
    pop = [create_individual(length) for _ in range(pop_size)]
    evaluate_population(pop)
    return pop

def evaluate(ind: Individual) -> int:
    ind.fitness = sum(ind.genes)
    return ind.fitness

def evaluate_population(pop: List[Individual]) -> None:
    for ind in pop:
        evaluate(ind)

def tournament_one(pop: List[Individual], k: int = TOUR_SIZE) -> Individual:
    contenders = random.sample(pop, k)
    best = max(contenders, key=lambda x: x.fitness)
    return best.copy()

def select_tournament(pop: List[Individual], n: int, k: int = TOUR_SIZE) -> List[Individual]:
    return [tournament_one(pop, k) for _ in range(n)]

def cx_one_point_inplace(a: Individual, b: Individual) -> None:
    if len(a.genes) < 2:
        return
    pt = random.randint(1, len(a.genes) - 1)
    a_tail = a.genes[pt:]
    b_tail = b.genes[pt:]
    a.genes[pt:] = b_tail
    b.genes[pt:] = a_tail
    a.fitness = None
    b.fitness = None

def mut_flip_bit_inplace(ind: Individual, indpb: float) -> None:
    for i in range(len(ind.genes)):
        if random.random() < indpb:
            ind.genes[i] = 1 - ind.genes[i]
    ind.fitness = None

def main_adaptive_ga(pop_size: int = POP_SIZE,
                     gens: int = GENERATIONS) -> Tuple[Individual, int]:
    pop = init_population(pop_size, GENE_LENGTH)
    best = max(pop, key=lambda x: x.fitness).copy()

    for gen in range(gens):
        # Selection
        offspring = select_tournament(pop, pop_size, k=TOUR_SIZE)

        # Crossover with adaptive Pc
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            # use best fitness of the pair to compute Pc_adapt similar to example
            pair_best = max(c1.fitness or evaluate(c1), c2.fitness or evaluate(c2))
            Pc_adapt = PC_MAX * (MAX_FIT - pair_best) / MAX_FIT
            Pc_adapt = max(0.0, min(1.0, Pc_adapt))
            if random.random() < Pc_adapt:
                cx_one_point_inplace(c1, c2)

        # Mutation with adaptive Pm per individual
        for ind in offspring:
            ind_f = ind.fitness if ind.fitness is not None else evaluate(ind)
            Pm_adapt = PM_BASE * (MAX_FIT - ind_f) / MAX_FIT
            Pm_adapt = max(0.0, min(1.0, Pm_adapt))
            if random.random() < Pm_adapt:
                # use a small per-gene prob (keeping behavior similar to DEAP mutFlipBit)
                mut_flip_bit_inplace(ind, indpb=0.01)

        # Re-evaluate population where needed
        evaluate_population(offspring)

        # Replace
        pop = offspring

        # Update best
        gen_best = max(pop, key=lambda x: x.fitness)
        if gen_best.fitness > best.fitness:
            best = gen_best.copy()

        # optional progress print
        if gen % 10 == 0 or gen == gens - 1:
            print(f"Gen {gen}: best_fitness={best.fitness}")

    return best, best.fitness

if __name__ == "__main__":
    best_ind, best_fit = main_adaptive_ga()
    print(f"Best fitness: {best_fit}/{MAX_FIT}")
    print(f"Best individual (first 100 bits): {best_ind.genes[:100]}")