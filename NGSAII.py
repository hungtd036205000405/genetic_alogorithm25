import random
from typing import List, Tuple

# GA params
IND_SIZE = 5
POP_SIZE = 100
GENERATIONS = 50
CX_PROB = 0.7
MUT_PROB = 0.2

class Individual:
    def __init__(self, genes: List[int]):
        self.genes = genes
        self.objectives = None  # [f1, f2]
        self.rank = None
        self.crowding_distance = 0.0

    def copy(self):
        ind = Individual(self.genes.copy())
        ind.objectives = self.objectives.copy() if self.objectives else None
        ind.rank = self.rank
        ind.crowding_distance = self.crowding_distance
        return ind

    def __repr__(self):
        return f"['{''.join(map(str, self.genes))}'] | f1={int(self.objectives[0])} | f2={int(self.objectives[1])}"

def create_individual(size: int = IND_SIZE) -> Individual:
    genes = [random.randint(0, 1) for _ in range(size)]
    return Individual(genes)

def init_population(pop_size: int = POP_SIZE, size: int = IND_SIZE) -> List[Individual]:
    return [create_individual(size) for _ in range(pop_size)]

def eval_bi_objective(individual: Individual) -> None:
    """
    Tính Fitness cho hai mục tiêu:
    f1: Tổng số bit 1 (tối đa hóa)
    f2: Tổng số bit 0 (tối đa hóa)
    """
    f1 = sum(individual.genes)  # Tổng số bit 1
    f2 = IND_SIZE - f1           # Tổng số bit 0
    individual.objectives = [f1, f2]

def evaluate_population(pop: List[Individual]) -> None:
    for ind in pop:
        if ind.objectives is None:
            eval_bi_objective(ind)

def dominates(a: Individual, b: Individual) -> bool:
    """Kiểm tra a chi phối b (Maximize cả 2 mục tiêu)"""
    a_better = all(a.objectives[i] >= b.objectives[i] for i in range(len(a.objectives)))
    b_better = all(b.objectives[i] > a.objectives[i] for i in range(len(a.objectives)))
    return a_better and b_better

def fast_non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    """Sắp xếp không bị chi phối nhanh cho NSGA-II"""
    fronts = [[]]
    for p in pop:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                p.dominated_solutions.append(q)
            elif dominates(q, p):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove empty last front

def calculate_crowding_distance(front: List[Individual]) -> None:
    """Tính toán khoảng cách tinh thể cho các cá thể trong front"""
    if len(front) == 0:
        return

    for ind in front:
        ind.crowding_distance = 0

    nobj = len(front[0].objectives)
    for m in range(nobj):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')

        if front[-1].objectives[m] - front[0].objectives[m] > 0:
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / \
                           (front[-1].objectives[m] - front[0].objectives[m])
                front[i].crowding_distance += distance

def cx_uniform(p1: Individual, p2: Individual, indpb: float = 0.5) -> Tuple[Individual, Individual]:
    """Uniform crossover: mỗi gene từ p1 hoặc p2 với xác suất indpb"""
    c1, c2 = Individual(p1.genes.copy()), Individual(p2.genes.copy())
    for i in range(len(p1.genes)):
        if random.random() < indpb:
            c1.genes[i], c2.genes[i] = c2.genes[i], c1.genes[i]
    c1.objectives = None
    c2.objectives = None
    return c1, c2

def mutate_flip_bit(ind: Individual, indpb: float = 0.05) -> Individual:
    """Mutation: lật bit với xác suất indpb"""
    for i in range(len(ind.genes)):
        if random.random() < indpb:
            ind.genes[i] = 1 - ind.genes[i]
    ind.objectives = None
    return ind

def tournament_select_nsga2(pop: List[Individual], k: int = 2) -> Individual:
    """Tournament selection dựa trên rank và crowding distance"""
    contenders = random.sample(pop, k)
    best = max(contenders, key=lambda x: (-x.rank, x.crowding_distance))
    return best.copy()

def main_nsga2_bi_objective(ngen: int = 50, mu: int = 100) -> List[Individual]:
    """
    Chạy thuật toán NSGA-II.
    ngen: Số thế hệ. mu: Kích thước quần thể.
    """
    pop = init_population(mu, IND_SIZE)
    evaluate_population(pop)

    # Tính rank ban đầu
    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        calculate_crowding_distance(front)

    for gen in range(ngen):
        # Selection & Variation
        offspring = []
        for _ in range(mu // 2):
            p1 = tournament_select_nsga2(pop)
            p2 = tournament_select_nsga2(pop)
            if random.random() < CX_PROB:
                c1, c2 = cx_uniform(p1, p2, indpb=0.5)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if random.random() < MUT_PROB:
                c1 = mutate_flip_bit(c1, indpb=0.05)
            if random.random() < MUT_PROB:
                c2 = mutate_flip_bit(c2, indpb=0.05)
            offspring.extend([c1, c2])

        # Evaluate offspring
        evaluate_population(offspring)

        # Combine parent + offspring
        combined = pop + offspring

        # Non-dominated sorting
        fronts = fast_non_dominated_sort(combined)

        # Calculate crowding distance
        for front in fronts:
            calculate_crowding_distance(front)

        # Select next generation
        pop = []
        for front in fronts:
            if len(pop) + len(front) <= mu:
                pop.extend(front)
            else:
                front.sort(key=lambda x: -x.crowding_distance)
                pop.extend(front[:mu - len(pop)])
                break

    # Lấy tập hợp các nghiệm Pareto Front cuối cùng
    fronts = fast_non_dominated_sort(pop)
    pareto_front = fronts[0] if len(fronts) > 0 else pop

    return pareto_front

if __name__ == "__main__":
    # Đặt Seed để đảm bảo kết quả có thể tái lập
    random.seed(42)

    # Chạy NSGA-II
    final_front = main_nsga2_bi_objective(ngen=50, mu=100)

    print("✨ --- Kết quả NSGA-II (L=5) --- ✨")
    print(f"Tổng số nghiệm Pareto: {len(final_front)}")
    print("\nChi tiết tập Pareto (Chuỗi | f1 | f2):")

    # In ra các cá thể trên tập Pareto Front
    for ind in final_front:
        print(ind)