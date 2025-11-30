import random
# Chọn cha mẹ
parent1 = tournament_selection(pop, fitnesses, k=3)
parent2 = tournament_selection(pop, fitnesses, k=3)


# Lai ghép
if random.random() < cfg.crossover_rate:
if cfg.representation == 'binary':
child1, child2 = one_point_crossover(parent1, parent2)
else:
child1, child2 = sbx_crossover(parent1, parent2)
else:
child1, child2 = parent1.copy(), parent2.copy()


# Đột biến
if cfg.representation == 'binary':
child1 = bitflip_mutation(child1, cfg.mutation_rate)
child2 = bitflip_mutation(child2, cfg.mutation_rate)
else:
child1 = gaussian_mutation(child1, cfg.mutation_rate, sigma=0.1, bounds=cfg.real_bounds)
child2 = gaussian_mutation(child2, cfg.mutation_rate, sigma=0.1, bounds=cfg.real_bounds)


new_pop.append(child1)
if len(new_pop) < cfg.population_size:
new_pop.append(child2)


pop = new_pop
fitnesses = [fitness_fn(ind) for ind in pop]


# Cập nhật best
gen_best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
gen_best_fit = fitnesses[gen_best_idx]
gen_best_ind = pop[gen_best_idx]


if gen_best_fit > best_fitness:
best_fitness = gen_best_fit
best_individual = gen_best_ind.copy()


if verbose and (gen % max(1, cfg.generations // 10) == 0 or gen == 1 or gen == cfg.generations):
print(f"Gen {gen:4d}/{cfg.generations} | Best fitness: {best_fitness:.6f}")


return best_individual, best_fitness


# ---------------------------
# Ví dụ minh họa: Tối đa hóa -Sphere- (chúng ta sẽ tối đa hóa -f(x) tức là tối thiểu f(x))
# f(x) = sum(x_i^2) -> tối ưu là 0; để phù hợp mẫu GA (maximize) ta dùng fitness = -f(x)
# ---------------------------


def sphere_fitness(ind: List[float]) -> float:
# Trả về fitness (càng lớn càng tốt). Ta muốn MINIMIZE sum(x^2), nên fitness = -sum(x^2)
return -sum(xi*xi for xi in ind)


if __name__ == '__main__':
random.seed(42)


cfg = GAConfig(population_size=60,
chromosome_length=10,
generations=200,
crossover_rate=0.9,
mutation_rate=0.05,
elitism=2,
representation='real',
real_bounds=(-5.0, 5.0))


best, best_fit = run_ga(cfg, sphere_fitness, verbose=True)


print('\nKET QUA:')
print('Best fitness:', best_fit)
print('Best individual:', best)