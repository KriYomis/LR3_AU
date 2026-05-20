import os, random
from dataclasses import dataclass
from PIL import Image


@dataclass
class GAConfig:
    population_size:  int   = 10  
    num_generations:  int   = 10000   
    mutation_chance:  float = 0.15
    mutation_chance_2: float = 0.30  
    weight_range:     float = 1.0   
    bias_range:  float = 5.0   

random.seed(42)
NUM_PIXELS = 81

def read_png(filepath: str) -> list:
    img = Image.open(filepath).convert('L').resize((9, 9))
    return [1.0 if img.getpixel((c, r)) < 128 else 0.0
            for r in range(9) for c in range(9)]


def load_dataset(folder: str) -> list:
    data = []
    for cls, lbl in {'plus': 1, 'V': 0}.items():
        path = os.path.join(folder, cls)
        for fname in sorted(os.listdir(path)):
            if fname.endswith('.png'):
                pixels = read_png(os.path.join(path, fname))
                data.append({'pixels': pixels, 'label': lbl, 'name': fname})
    return data


def neuron(weights: list, bias: float, pixels: list) -> int:
    total = sum(weights[i] * pixels[i] for i in range(NUM_PIXELS))
    return 1 if total >= -bias else 0

def fitness(ind: dict, dataset: list) -> float:
    correct = sum(
        1 for s in dataset
        if neuron(ind['weights'], ind['bias'], s['pixels']) == s['label']
    )
    return correct / len(dataset)



def make_individual(cfg: GAConfig) -> dict:
    return {
        'weights':   [random.uniform(-1.0, 1.0)
                      for _ in range(NUM_PIXELS)],
        'bias': random.uniform(-cfg.bias_range, cfg.bias_range),
        'fitness':   0.0
    }


def crossover(a: dict, b: dict) -> dict:
    cut = random.randint(1, NUM_PIXELS - 1)
    return {
        'weights':   a['weights'][:cut] + b['weights'][cut:],
        'bias': random.choice([a['bias'], b['bias']]),
        'fitness':   0.0
    }


def mutate(ind: dict, cfg: GAConfig) -> dict:
    w = ind['weights'][:]
    if random.random() < cfg.mutation_chance:
        p1 = random.randint(0, NUM_PIXELS - 1)
        p2 = random.randint(0, NUM_PIXELS - 1)
        while p2 == p1:
            p2 = random.randint(0, NUM_PIXELS - 1)
        w[p1], w[p2] = w[p2], w[p1]
    if random.random() < cfg.mutation_chance_2:
        p1 = random.randint(0, NUM_PIXELS - 1)
        w[p1] += random.uniform(-0.3, 0.3)

    return {'weights': w, 'bias': ind['bias'], 'fitness': 0.0}


def training(cfg: GAConfig, train: list) -> dict:

    population = [make_individual(cfg) for _ in range(cfg.population_size)]
    for ind in population:
        ind['fitness'] = fitness(ind, train)

    print(f"{'Поколение':>10}  {'Fitness':>8} {'Первые 4 веса':>20}")

    for gen in range(1, cfg.num_generations + 1):

        population.sort(key=lambda x: x['fitness'], reverse=True)
        best = population[0]

        if gen == 1 or gen % 10 == 0:
            print(f"{gen:>10}  {best['fitness']:>8} {', '.join(f'{w:.4f}' for w in best['weights'][:4])}")


        if best['fitness'] == 1.0:
            print(f"\nДостигнута 100% точность на поколении {gen}!")
            print(f"Лучший нейрон: порог {best['bias']:.4f}, первые 4 веса {best['weights'][:4]}")
            break

        new_pop = [best]

        while len(new_pop) < cfg.population_size:
            pa    = random.choice(population)   
            pb    = random.choice(population)   
            child = crossover(pa, pb)           
            child = mutate(child, cfg)     
            child['fitness'] = fitness(child, train)
            new_pop.append(child)

        population = new_pop

    population.sort(key=lambda x: x['fitness'], reverse=True)
    return population[0]


def predict_image(best: dict, filepath: str) -> None:
    pixels = read_png(filepath)
    result = neuron(best['weights'], best['bias'], pixels)
    label  = '+ (плюс)' if result == 1 else 'V (галочка)'
    print(f"\nИзображение: {os.path.basename(filepath)}")
    for r in range(9):
        row = ''.join('  ' if pixels[r*9+c] else '##' for c in range(9))
        print(f"|{row}|")
    print(f"\nОтвет нейрона: {label}")



cfg = GAConfig()


print(f"  Популяция : {cfg.population_size}")
print(f"  Поколений : {cfg.num_generations}")
print(f"  Мутация   : {cfg.mutation_chance:.0%}")

train = load_dataset('train')
print(f"\nЗагружено обучающих примеров: {len(train)}")

best = training(cfg, train)

print(f"\nЛучший нейрон:")
print(f"Фитнес функция : {best['fitness']}")
print(f"Порог th : {best['bias']:.4f}")


print(f"\nДЕМОНСТРАЦИЯ\n")
while True:
    path = input("Путь к изображению: ").strip()
    if path.lower() == 'q':
        break
    if not os.path.isfile(path):
        print(f"Файл не найден: {path}")
        continue
    predict_image(best, path)