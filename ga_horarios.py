import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# -----------------------------
# Configuración básica del problema
# -----------------------------

DAYS = ["Lun", "Mar", "Mié", "Jue", "Vie"]
SLOTS_PER_DAY = 6
D = len(DAYS)
T = SLOTS_PER_DAY


@dataclass
class Course:
    name: str
    teacher: str
    room: str
    sessions_per_week: int  # cuántos bloques a la semana


# Ejemplo sencillo de cursos
COURSES: List[Course] = [
    Course("Matemáticas", "Prof. A", "Aula 1", 3),
    Course("Programación", "Prof. B", "Aula 2", 3),
    Course("Física", "Prof. A", "Aula 3", 2),
    Course("Historia", "Prof. C", "Aula 1", 2),
]


# -----------------------------
# Evaluación de factibilidad y calidad
# -----------------------------

def check_feasibility(individual: List[np.ndarray]) -> int:
    """
    Verifica si un horario es factible:
    - Sin choques de profesor
    - Sin choques de aula
    Regresa 1 si es factible, 0 si NO lo es.
    """
    teacher_occ = {}
    room_occ = {}

    for mat, course in zip(individual, COURSES):
        if course.teacher not in teacher_occ:
            teacher_occ[course.teacher] = np.zeros((D, T), dtype=int)
        if course.room not in room_occ:
            room_occ[course.room] = np.zeros((D, T), dtype=int)
        teacher_occ[course.teacher] += mat
        room_occ[course.room] += mat

    # Si alguna celda > 1, hay choque
    for occ in list(teacher_occ.values()) + list(room_occ.values()):
        if (occ > 1).any():
            return 0

    return 1


def count_gaps_for_course(mat: np.ndarray) -> int:
    """
    Cuenta 'huecos' dentro de un día para un curso.
    Un hueco es un bloque vacío entre bloques ocupados en el mismo día.
    """
    gaps = 0
    for d in range(D):
        row = mat[d, :]
        ones = np.where(row == 1)[0]
        if len(ones) > 1:
            first, last = ones[0], ones[-1]
            # Recorremos entre el primero y el último encendido
            for t in range(first, last):
                if row[t] == 0 and row[t + 1] == 1:
                    gaps += 1
    return gaps


def fitness(individual: List[np.ndarray],
            penalize_gaps: bool = False) -> float:
    """
    Calcula el fitness de un individuo.
    - Si no es factible (choques), fitness casi 0.
    - Si es factible:
        * Se premia usar varios días (mayor cobertura).
        * Opcionalmente se penalizan huecos.
    """
    feasible = check_feasibility(individual)
    if feasible == 0:
        return 0.001

    total_sessions = 0
    days_used = set()
    total_gaps = 0

    for mat, course in zip(individual, COURSES):
        sessions = int(mat.sum())
        total_sessions += sessions

        # Días en que se imparte esta materia
        for d in range(D):
            if mat[d, :].sum() > 0:
                days_used.add(d)
        if penalize_gaps:
            total_gaps += count_gaps_for_course(mat)

    if total_sessions == 0:
        return 0.001

    day_coverage = len(days_used) / D  # entre 0 y 1

    if penalize_gaps:
        gap_penalty = 1 / (1 + total_gaps)
    else:
        gap_penalty = 1.0

    return day_coverage * gap_penalty


# -----------------------------
# Generación inicial y reparación
# -----------------------------

def initialize_population(pop_size: int) -> List[List[np.ndarray]]:
    """
    Genera una población inicial de horarios aleatorios.
    Cada curso se asigna a 'sessions_per_week' bloques al azar.
    """
    population = []
    for _ in range(pop_size):
        individual = []
        for course in COURSES:
            mat = np.zeros((D, T), dtype=int)
            # elegimos aleatoriamente los bloques de este curso
            slots = [(d, t) for d in range(D) for t in range(T)]
            random.shuffle(slots)
            for (d, t), _ in zip(slots, range(course.sessions_per_week)):
                mat[d, t] = 1
            individual.append(mat)
        population.append(individual)
    return population


def repair_sessions(mat: np.ndarray, sessions_needed: int) -> None:
    """
    Repara el número de sesiones de una matriz de curso, para que
    tenga exactamente 'sessions_needed' bloques encendidos.
    """
    ones = np.argwhere(mat == 1)
    current = len(ones)

    if current > sessions_needed:
        # Apagar algunos
        to_off = current - sessions_needed
        idxs = list(range(current))
        random.shuffle(idxs)
        for j in idxs[:to_off]:
            d, t = ones[j]
            mat[d, t] = 0
    elif current < sessions_needed:
        # Encender algunos
        zeros = np.argwhere(mat == 0)
        random.shuffle(zeros)
        needed = sessions_needed - current
        for d, t in zeros:
            if mat[d, t] == 0:
                mat[d, t] = 1
                needed -= 1
                if needed <= 0:
                    break


# -----------------------------
# Selección, cruza y mutación
# -----------------------------

def roulette_selection(population: List[List[np.ndarray]],
                       fits: List[float]) -> List[np.ndarray]:
    """Selección proporcional al fitness (tipo ruleta)."""
    total_fit = sum(fits)
    if total_fit == 0:
        return random.choice(population)

    r = random.random() * total_fit
    acc = 0.0
    for indiv, f in zip(population, fits):
        acc += f
        if acc >= r:
            return [m.copy() for m in indiv]

    # Fallback
    return [m.copy() for m in population[-1]]


def crossover(
    p1: List[np.ndarray],
    p2: List[np.ndarray],
    crossover_rate: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Cruza multipunto sencilla: intercambia días completos entre padres
    para cada curso.
    """
    if random.random() > crossover_rate:
        return [m.copy() for m in p1], [m.copy() for m in p2]

    c1 = []
    c2 = []
    for m1, m2, course in zip(p1, p2, COURSES):
        child1 = m1.copy()
        child2 = m2.copy()
        # Para cada día decidimos si intercambiamos
        for d in range(D):
            if random.random() < 0.5:
                child1[d, :], child2[d, :] = child2[d, :].copy(), child1[d, :].copy()
        # Reparamos número de sesiones
        repair_sessions(child1, course.sessions_per_week)
        repair_sessions(child2, course.sessions_per_week)
        c1.append(child1)
        c2.append(child2)
    return c1, c2


def mutate(individual: List[np.ndarray], mutation_rate: float) -> None:
    """Mutación: intercambia bloques al azar y repara sesiones."""
    for i, course in enumerate(COURSES):
        if random.random() < mutation_rate:
            mat = individual[i]
            d1, t1 = random.randint(0, D - 1), random.randint(0, T - 1)
            d2, t2 = random.randint(0, D - 1), random.randint(0, T - 1)
            mat[d1, t1], mat[d2, t2] = mat[d2, t2], mat[d1, t1]
            repair_sessions(mat, course.sessions_per_week)


# -----------------------------
# Impresión de horarios
# -----------------------------

def print_schedule(individual: List[np.ndarray]) -> None:
    """Imprime el horario de cada curso de forma sencilla."""
    for mat, course in zip(individual, COURSES):
        print("\nCurso:", course.name)
        print("Profesor:", course.teacher, "| Aula:", course.room)
        print("Horario:")
        print("     " + " ".join([f"{i:^3}" for i in range(T)]))
        for d in range(D):
            row = " ".join([" X " if mat[d, t] == 1 else " . "
                            for t in range(T)])
            print(f"{DAYS[d]:>3}: {row}")


# -----------------------------
# Bucle principal del AG
# -----------------------------

def run_ga(population_size: int,
           generations: int,
           crossover_rate: float,
           mutation_rate: float,
           penalize_gaps: bool = False,
           use_adaptive_mutation: bool = False,
           label: str = "GA"):
    """
    Ejecuta el Algoritmo Genético y regresa el mejor individuo y su fitness.
    """
    print(f"\n=== Ejecutando {label} ===")
    population = initialize_population(population_size)
    best_ind = None
    best_fit = -1.0
    current_mutation = mutation_rate

    for gen in range(generations):
        fits = [fitness(ind, penalize_gaps=penalize_gaps) for ind in population]
        gen_best_idx = int(np.argmax(fits))
        gen_best_fit = fits[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = [m.copy() for m in population[gen_best_idx]]

        # Mutación adaptativa simple (solo en la versión mejorada)
        if use_adaptive_mutation and gen > 0 and gen % 100 == 0:
            current_mutation = min(current_mutation + 0.01, 0.2)

        # Mostrar progreso cada 100 generaciones
        if gen % 100 == 0:
            print(f"Gen {gen:4d} | mejor fitness: {gen_best_fit:.4f} | mutación: {current_mutation:.3f}")

        # Nueva población
        new_pop = []
        while len(new_pop) < population_size:
            p1 = roulette_selection(population, fits)
            p2 = roulette_selection(population, fits)
            c1, c2 = crossover(p1, p2, crossover_rate)
            mutate(c1, current_mutation)
            mutate(c2, current_mutation)
            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        population = new_pop

    print(f"\n{label} - mejor fitness global: {best_fit:.4f}")
    print_schedule(best_ind)
    return best_ind, best_fit


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Versión base (según configuración del reporte)
    base_ind, base_fit = run_ga(
        population_size=50,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.05,
        penalize_gaps=False,
        use_adaptive_mutation=False,
        label="Algoritmo Base"
    )

    # Versión mejorada (población 80, mutación adaptativa, penalización de huecos)
    improved_ind, improved_fit = run_ga(
        population_size=80,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.05,
        penalize_gaps=True,
        use_adaptive_mutation=True,
        label="Algoritmo Mejorado"
    )

    print("\n=== Comparación final ===")
    print(f"Mejor fitness (base):      {base_fit:.4f}")
    print(f"Mejor fitness (mejorado):  {improved_fit:.4f}")
