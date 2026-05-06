"""
Обучение одного нейрона через Генетический Алгоритм
Датасет: папка train/plus/ и train/v/  (PNG, любой размер)
После обучения: вводим путь к изображению — нейрон его классифицирует.

pip install pillow   <- только для чтения PNG
"""

import os, random
from dataclasses import dataclass
from PIL import Image


# ─────────────────────────────────────────────────────────
#  ПАРАМЕТРЫ  (все настройки алгоритма в одном месте)
# ─────────────────────────────────────────────────────────

@dataclass
class GAConfig:
    population_size:  int   = 40    # количество особей в популяции
    num_generations:  int   = 100   # сколько поколений эволюции
    mutation_chance:  float = 0.15  # вероятность мутации  (0.0 – 1.0)
    weight_range:     float = 1.0   # веса инициализируются в [-X, +X]
    threshold_range:  float = 5.0   # порог инициализируется в [-X, +X]
NUM_PIXELS = 81

# ─────────────────────────────────────────────────────────
#  ЧТЕНИЕ PNG  (не алгоритм, просто утилита)
#
#  PNG любого размера масштабируется до 9x9.
#  Тёмный пиксель (< 128) -> 1.0,  светлый -> 0.0
# ─────────────────────────────────────────────────────────

  # 9 x 9

def read_png(filepath: str) -> list:
    """PNG -> список из 81 числа (0.0 или 1.0)."""
    img = Image.open(filepath).convert('L').resize((9, 9))
    return [1.0 if img.getpixel((c, r)) < 128 else 0.0
            for r in range(9) for c in range(9)]


def load_dataset(folder: str) -> list:
    """
    Читает train/plus/*.png  -> label=1  (класс «+»)
             train/v/*.png    -> label=0  (класс «V»)
    Возвращает список словарей: {'pixels', 'label', 'name'}
    """
    data = []
    for cls, lbl in {'plus': 1, 'V': 0}.items():
        path = os.path.join(folder, cls)
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith('.png'):
                pixels = read_png(os.path.join(path, fname))
                data.append({'pixels': pixels, 'label': lbl, 'name': fname})
    return data


# ─────────────────────────────────────────────────────────
#  НЕЙРОН
#
#  S = w[0]*x[0] + w[1]*x[1] + ... + w[80]*x[80]
#  если S >= θ  ->  выход 1  (класс «+»)
#  если S <  θ  ->  выход 0  (класс «V»)
# ─────────────────────────────────────────────────────────

def neuron(weights: list, threshold: float, pixels: list) -> int:
    total = sum(weights[i] * pixels[i] for i in range(NUM_PIXELS))
    return 1 if total >= threshold else 0


# ─────────────────────────────────────────────────────────
#  ФИТНЕС-ФУНКЦИЯ
#
#  fitness = правильных ответов / всего примеров   (0.0 – 1.0)
#
#  Визуальная шкала: чем больше X — тем особь приспособленнее.
#  Цель алгоритма — максимизировать fitness до 1.0 (100%).
# ─────────────────────────────────────────────────────────

def fitness(ind: dict, dataset: list) -> float:
    correct = sum(
        1 for s in dataset
        if neuron(ind['weights'], ind['threshold'], s['pixels']) == s['label']
    )
    return correct / len(dataset)


def fitness_bar(value: float, width: int = 20) -> str:
    """[XXXXXXXXXXXXXXX.....] 75%"""
    filled = round(value * width)
    return f"[{'X' * filled + '.' * (width - filled)}] {value:.0%}"


# ─────────────────────────────────────────────────────────
#  ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
# ─────────────────────────────────────────────────────────

def make_individual(cfg: GAConfig) -> dict:
    """Случайная особь: случайные веса и порог."""
    return {
        'weights':   [random.uniform(-cfg.weight_range, cfg.weight_range)
                      for _ in range(NUM_PIXELS)],
        'threshold': random.uniform(-cfg.threshold_range, cfg.threshold_range),
        'fitness':   0.0
    }


def crossover(a: dict, b: dict) -> dict:
    """
    Скрещивание через случайную точку разреза.
    Потомок = [гены A до точки] + [гены B после точки].
    Порог наследуется случайно от одного из родителей.
    """
    cut = random.randint(1, NUM_PIXELS - 1)
    return {
        'weights':   a['weights'][:cut] + b['weights'][cut:],
        'threshold': random.choice([a['threshold'], b['threshold']]),
        'fitness':   0.0
    }


def mutate(ind: dict, cfg: GAConfig) -> dict:
    """
    Мутация: выбираем 2 случайные позиции и меняем их местами (swap).
    Применяется с вероятностью mutation_chance.
    """
    w = ind['weights'][:]
    if random.random() < cfg.mutation_chance:
        p1 = random.randint(0, NUM_PIXELS - 1)
        p2 = random.randint(0, NUM_PIXELS - 1)
        while p2 == p1:
            p2 = random.randint(0, NUM_PIXELS - 1)
        w[p1], w[p2] = w[p2], w[p1]  # swap
    return {'weights': w, 'threshold': ind['threshold'], 'fitness': 0.0}


def run(cfg: GAConfig, train: list) -> dict:
    """Основной цикл ГА. Возвращает лучшую особь."""

    # ШАГ 1 — Инициализация случайной популяции
    population = [make_individual(cfg) for _ in range(cfg.population_size)]
    for ind in population:
        ind['fitness'] = fitness(ind, train)

    print(f"\n{'─'*46}")
    print(f"  {'Поколение':>10}  {'Fitness':>8}")
    print(f"{'─'*46}")

    for gen in range(1, cfg.num_generations + 1):

        # ШАГ 2 — Сортируем: лучшие особи идут первыми
        population.sort(key=lambda x: x['fitness'], reverse=True)
        best = population[0]

        if gen == 1 or gen % 10 == 0:
            print(f"  {gen:>10}  {best['fitness']}")

        if best['fitness'] >= 1.0:
            print(f"\n  Достигнута 100% точность на поколении {gen}!")
            break

        # ШАГ 3 — Новое поколение
        new_pop = [best]  # элитизм: лучшая особь выживает без изменений

        while len(new_pop) < cfg.population_size:
            pa    = random.choice(population)   # случайный родитель A
            pb    = random.choice(population)   # случайный родитель B
            child = crossover(pa, pb)           # скрещивание
            child = mutate(child, cfg)          # мутация
            child['fitness'] = fitness(child, train)
            new_pop.append(child)

        population = new_pop

    population.sort(key=lambda x: x['fitness'], reverse=True)
    return population[0]


# ─────────────────────────────────────────────────────────
#  ДЕМОНСТРАЦИЯ — классификация одного изображения
# ─────────────────────────────────────────────────────────

def predict_image(best: dict, filepath: str) -> None:
    """Читает PNG по пути и выводит ответ нейрона."""
    pixels = read_png(filepath)
    result = neuron(best['weights'], best['threshold'], pixels)
    label  = '+ (плюс)' if result == 1 else 'V (галочка)'

    # Показываем изображение в терминале символами
    print(f"\n  Изображение: {os.path.basename(filepath)}")
    print("  " + "─" * 19)
    for r in range(9):
        row = ''.join('##' if pixels[r*9+c] else '  ' for c in range(9))
        print(f"  |{row}|")
    print("  " + "─" * 19)
    print(f"\n  Ответ нейрона: {label}")


# ─────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────

def main():
    cfg = GAConfig(
        population_size = 20,
        num_generations = 100,
        mutation_chance = 0.15,
    )

    print("=" * 46)
    print("  Нейрон + Генетический Алгоритм")
    print("  Классы: «+» и «V»,  PNG 9x9")
    print("=" * 46)
    print(f"\n  Популяция : {cfg.population_size}")
    print(f"  Поколений : {cfg.num_generations}")
    print(f"  Мутация   : {cfg.mutation_chance:.0%}")

    # Загружаем обучающий датасет из папки train/
    train = load_dataset('train')
    print(f"\n  Загружено обучающих примеров: {len(train)}")

    # Запускаем обучение
    best = run(cfg, train)

    print(f"\n  Лучший нейрон:")
    print(f"    Фитнес функция : {best['fitness']}")
    print(f"    Порог th : {best['threshold']:.4f}")

    # Демонстрация: пользователь вводит путь к PNG
    print(f"\n{'─'*46}")
    print("  ДЕМОНСТРАЦИЯ")
    print("  Введите путь к PNG-файлу (или 'q' для выхода):")
    print(f"{'─'*46}")

    while True:
        path = input("\n  Путь к изображению: ").strip()
        if path.lower() == 'q':
            break
        if not os.path.isfile(path):
            print(f"  Файл не найден: {path}")
            continue
        predict_image(best, path)


if __name__ == '__main__':
    random.seed(38)
    main()