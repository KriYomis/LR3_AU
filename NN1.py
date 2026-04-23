import os
from PIL import Image

from GA2 import GeneticAlgorithm, GeneticAlgorithmConfig


DATASETS = ("pictures", "more", "mix")
LABELS = {"+": 1, "V": 0}


def load_image(path):
    image = Image.open(path).convert("L")
    return [1.0 if pixel >= 128 else 0.0 for pixel in image.tobytes()]


def load_dataset(folder):
    X = []
    y = []
    for class_name, label in LABELS.items():
        class_folder = os.path.join(folder, class_name)
        for filename in sorted(os.listdir(class_folder)):
            if filename.endswith(".png"):
                X.append(load_image(os.path.join(class_folder, filename)))
                y.append(label)
    return X, y


def predict(weights, bias, x):
    score = bias
    for i in range(len(x)):
        score += weights[i] * x[i]
    return 1 if score > 0 else 0


def fitness(genome, X, y):
    weights = genome[:-1]
    bias = genome[-1]
    total = 0.0
    for x, target in zip(X, y):
        total += 1.0 if predict(weights, bias, x) == target else -1.0
    return total


def train(X, y, generations):
    config = GeneticAlgorithmConfig(
        genome_length=len(X[0]) + 1,
        population_size=60,
        generations=generations,
        elite_size=6,
        tournament_size=4,
        mutation_rate=0.15,
        mutation_strength=0.4,
        crossover_rate=0.7,
        min_value=-3.0,
        max_value=3.0,
        random_seed=42,
    )
    algorithm = GeneticAlgorithm(
        config=config,
        fitness_function=lambda genome: fitness(genome, X, y),
    )

    def show_progress(trace):
        if trace.generation == 0:
            print("Первая популяция:")
            for i, genome in enumerate(trace.population, start=1):
                print(f"{i}: {[round(value, 3) for value in genome]}")
            print("Первая фитнесс функция:")
            print([round(value, 3) for value in trace.fitness_before])
            print()
            return True

        print(f"Поколение {trace.generation}")
        print("Фитнесс до редукции:")
        print([round(value, 3) for value in trace.fitness_before])
        print("Фитнесс после редукции:")
        print([round(value, 3) for value in trace.fitness_after])
        print("Лучшая особь после редукции:")
        print([round(value, 3) for value in trace.best_genome])
        print()
        return True

    best_genome, best_fitness, _ = algorithm.run(progress_callback=show_progress)
    return best_genome[:-1], best_genome[-1], best_fitness


def accuracy(weights, bias, X, y):
    correct = 0
    for x, target in zip(X, y):
        if predict(weights, bias, x) == target:
            correct += 1
    return correct / len(X)

train_name = "more"
generations = 20

train_X, train_y = load_dataset(train_name)
weights, bias, best_fitness = train(train_X, train_y, generations)

print(f"Обучение: {train_name}")
print(f"Поколений: {generations}")
print(f"\nЛучшая приспособленность: {best_fitness:.1f}")
for name in DATASETS:
    X, y = load_dataset(name)
    print(f"{name}: {accuracy(weights, bias, X, y) * 100:.1f}%")
