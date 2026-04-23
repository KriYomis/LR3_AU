from dataclasses import dataclass
import random


@dataclass
class GeneticAlgorithmConfig:
    genome_length: int
    population_size: int = 40
    generations: int = 80
    elite_size: int = 4
    tournament_size: int = 3
    mutation_rate: float = 0.12
    mutation_strength: float = 0.35
    crossover_rate: float = 0.85
    min_value: float = -2.0
    max_value: float = 2.0
    random_seed: int = 42


@dataclass
class GenerationResult:
    generation: int
    best_fitness: float
    average_fitness: float


@dataclass
class GenerationTrace:
    generation: int
    operation: str
    population: list
    fitness_before: list
    fitness_after: list
    best_genome: list
    best_fitness: float
    average_fitness: float
    details: list


class GeneticAlgorithm:
    def __init__(self, config, fitness_function):
        self.config = config
        self.fitness_function = fitness_function
        self.random = random.Random(config.random_seed)

    def _create_genome(self):
        return [
            self.random.uniform(self.config.min_value, self.config.max_value)
            for _ in range(self.config.genome_length)
        ]

    def _create_population(self):
        return [self._create_genome() for _ in range(self.config.population_size)]

    def _evaluate_population(self, population):
        scored_population = []
        for genome in population:
            fitness = self.fitness_function(genome)
            scored_population.append((fitness, genome))
        scored_population.sort(key=lambda item: item[0], reverse=True)
        return scored_population

    def _tournament_select(self, scored_population):
        candidates = self.random.sample(
            scored_population,
            min(self.config.tournament_size, len(scored_population)),
        )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _reduce_population(self, population):
        scored_population = self._evaluate_population(population)
        reduced = [genome[:] for _, genome in scored_population[: self.config.population_size]]
        reduced_scores = self._evaluate_population(reduced)
        return reduced, reduced_scores

    def _crossover(self, parent_a, parent_b):
        if self.random.random() > self.config.crossover_rate:
            return parent_a[:], parent_b[:]

        child_a = []
        child_b = []
        for gene_a, gene_b in zip(parent_a, parent_b):
            mix_ratio = self.random.random()
            child_a.append(gene_a * mix_ratio + gene_b * (1.0 - mix_ratio))
            child_b.append(gene_b * mix_ratio + gene_a * (1.0 - mix_ratio))
        return child_a, child_b

    def _mutate(self, genome):
        mutated = genome[:]
        changes = []
        for index, value in enumerate(mutated):
            if self.random.random() < self.config.mutation_rate:
                old_value = value
                value += self.random.uniform(
                    -self.config.mutation_strength,
                    self.config.mutation_strength,
                )
                value = max(self.config.min_value, min(self.config.max_value, value))
                mutated[index] = value
                changes.append((index, old_value, value))
        return mutated, changes

    def run(self, progress_callback=None):
        population = self._create_population()
        history = []

        if progress_callback is not None:
            initial_scores = self._evaluate_population(population)
            initial_fitness = [fitness for fitness, _ in initial_scores]
            initial_best_fitness, initial_best_genome = initial_scores[0]
            initial_average_fitness = sum(initial_fitness) / len(initial_fitness)
            should_continue = progress_callback(
                GenerationTrace(
                    generation=0,
                    operation="начальная популяция",
                    population=[genome[:] for genome in population],
                    fitness_before=initial_fitness,
                    fitness_after=initial_fitness[:],
                    best_genome=initial_best_genome[:],
                    best_fitness=initial_best_fitness,
                    average_fitness=initial_average_fitness,
                    details=[],
                )
            )
            if should_continue is False:
                return initial_best_genome[:], initial_best_fitness, history

        for generation in range(1, self.config.generations + 1):
            scored_population = self._evaluate_population(population)
            fitness_values = [fitness for fitness, _ in scored_population]
            best_fitness, _ = scored_population[0]
            average_fitness = sum(fitness_values) / len(fitness_values)
            working_population = [genome[:] for _, genome in scored_population]
            details = []

            if self.random.random() < self.config.crossover_rate:
                operation = "скрещивание"
                parent_a = self._tournament_select(scored_population)
                parent_b = self._tournament_select(scored_population)
                child_a, child_b = self._crossover(parent_a, parent_b)
                working_population.append(child_a)
                working_population.append(child_b)
                details.append("Выбрано действие: скрещивание")
                details.append(f"Родитель A fitness: {self.fitness_function(parent_a):.3f}")
                details.append(f"Родитель B fitness: {self.fitness_function(parent_b):.3f}")
                details.append(
                    f"Потомок A fitness: {self.fitness_function(child_a):.3f}"
                )
                details.append(
                    f"Потомок B fitness: {self.fitness_function(child_b):.3f}"
                )
            else:
                operation = "мутация"
                parent = self._tournament_select(scored_population)
                mutant, changes = self._mutate(parent)
                if not changes:
                    mutant_index = self.random.randrange(len(mutant))
                    old_value = mutant[mutant_index]
                    mutant[mutant_index] = max(
                        self.config.min_value,
                        min(
                            self.config.max_value,
                            old_value
                            + self.random.uniform(
                                -self.config.mutation_strength,
                                self.config.mutation_strength,
                            ),
                        ),
                    )
                    changes = [(mutant_index, old_value, mutant[mutant_index])]
                working_population.append(mutant)
                details.append("Выбрано действие: мутация")
                details.append(f"Исходная особь fitness: {self.fitness_function(parent):.3f}")
                details.append(
                    "Изменённые гены: "
                    + ", ".join(
                        f"{index + 1}: {old_value:.3f}->{new_value:.3f}"
                        for index, old_value, new_value in changes
                    )
                )
                details.append(
                    f"Мутировавшая особь fitness: {self.fitness_function(mutant):.3f}"
                )

            population, reduced_scores = self._reduce_population(working_population)
            reduced_fitness = [fitness for fitness, _ in reduced_scores]
            reduced_best_fitness, best_genome = reduced_scores[0]
            reduced_average_fitness = sum(reduced_fitness) / len(reduced_fitness)

            history.append(
                GenerationResult(
                    generation=generation,
                    best_fitness=reduced_best_fitness,
                    average_fitness=reduced_average_fitness,
                )
            )

            if progress_callback is not None:
                should_continue = progress_callback(
                    GenerationTrace(
                        generation=generation,
                        operation=operation,
                        population=[genome[:] for genome in population],
                        fitness_before=fitness_values,
                        fitness_after=reduced_fitness,
                        best_genome=best_genome[:],
                        best_fitness=reduced_best_fitness,
                        average_fitness=reduced_average_fitness,
                        details=details,
                    )
                )
                if should_continue is False:
                    final_population = self._evaluate_population(population)
                    best_fitness, best_genome = final_population[0]
                    return best_genome[:], best_fitness, history

        final_population = self._evaluate_population(population)
        best_fitness, best_genome = final_population[0]
        return best_genome[:], best_fitness, history
