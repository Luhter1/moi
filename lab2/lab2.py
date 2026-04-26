import numpy as np
import pandas as pd
from scipy import integrate

np.random.seed(42)

a, b = 2, 5 # пределы интегрирования
f = lambda x: x**2 # подынтегральная функция
N_list = [100, 1000, 10000, 100000] # число выборок

# Аналитическое значение
I_true, _ = integrate.quad(f, a, b)

print(f"Аналитическое значение интеграла: I = {I_true:.6f}")
print("=" * 70)


def relative_error(I_est, I_true):
    """Относительная погрешность"""
    return abs(I_est - I_true) / abs(I_true) * 100


def std_error(samples):
    """Оценка стандартного отклонения"""
    n = len(samples)
    return np.std(samples, ddof=1) / np.sqrt(n)


# ПРОСТОЙ МЕТОД МОНТЕ-КАРЛО
def mc_simple(f, a, b, N):
    X = np.random.uniform(a, b, N)
    samples = (b - a) * f(X)
    I_est = np.mean(samples)
    sigma = std_error(samples)
    return I_est, sigma


print("\n=== 1. Простой метод Монте-Карло ===")
rows = []
for N in N_list:
    I_est, sigma = mc_simple(f, a, b, N)
    err = relative_error(I_est, I_true)
    rows.append({"N": N,
                 "I_true": round(I_true, 6),
                 "I_est": round(I_est, 6),
                 "Погрешность, %": round(err, 4),
                 "sigma": round(sigma, 6)})
df1 = pd.DataFrame(rows)
print(df1.to_string(index=False))


# СТРАТИФИЦИРОВАННЫЙ МЕТОД МОНТЕ-КАРЛО
# Разбиваем [a,b] на M страт одинаковой длины
# В каждой страте берём N//M точек
def mc_stratified(f, a, b, N, step):
    strata_edges = np.arange(a, b + step * 0.5, step)
    M = len(strata_edges) - 1
    n_per = max(1, N // M)

    I_est = 0.0
    var_est = 0.0
    for j in range(M):
        a_j, b_j = strata_edges[j], strata_edges[j + 1]
        h_j = b_j - a_j
        X_j = np.random.uniform(a_j, b_j, n_per)
        fX = f(X_j)
        I_est += h_j * np.mean(fX)
        var_est += (h_j ** 2) * np.var(fX, ddof=1) / n_per

    sigma = np.sqrt(var_est)
    return I_est, sigma


print("\n=== 2. Стратифицированный метод Монте-Карло ===")
for step, label in [(1, "шаг=1"), (0.5, "шаг=0.5")]:
    print(f"\n  Разбиение с {label}")
    rows = []
    for N in N_list:
        I_est, sigma = mc_stratified(f, a, b, N, step)
        err = relative_error(I_est, I_true)
        rows.append({"N": N,
                     "I_true": round(I_true, 6),
                     "I_est": round(I_est, 6),
                     "Погрешность, %": round(err, 4),
                     "sigma": round(sigma, 6)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


# МЕТОД МОНТЕ-КАРЛО С ВЫБОРКОЙ ПО ЗНАЧИМОСТИ
C1 = 1.0 / (b - a)
C2 = 1.0 / (b**2 / 2 - a**2 / 2)
C3 = 1.0 / ((b**3 - a**3) / 3)


def sample_p1(N):
    """X ~ Uniform(a,b)"""
    return np.random.uniform(a, b, N)

def p1(x):
    return np.full_like(x, C1, dtype=float)


def sample_p2(N):
    U = np.random.uniform(0, 1, N)
    return np.sqrt(U * (b**2 - a**2) + a**2)

def p2(x):
    return C2 * x


def sample_p3(N):
    U = np.random.uniform(0, 1, N)
    return (U * (b**3 - a**3) + a**3) ** (1.0 / 3.0)

def p3(x):
    return C3 * x**2


def mc_importance(f, sampler, pdf, N):
    X = sampler(N)
    w = f(X) / pdf(X)
    I_est = np.mean(w)
    sigma = std_error(w)
    return I_est, sigma


print("\n=== 3. Выборка по значимости ===")
configs = [
    ("p1 (равномерная)", sample_p1, p1),
    ("p2 (линейная)", sample_p2, p2),
    ("p3 (квадратичная)", sample_p3, p3),
]
for name, sampler, pdf in configs:
    print(f"\n  Плотность: {name}")
    rows = []
    for N in N_list:
        I_est, sigma = mc_importance(f, sampler, pdf, N)
        err = relative_error(I_est, I_true)
        rows.append({"N": N,
                     "I_true": round(I_true, 6),
                     "I_est": round(I_est, 6),
                     "Погрешность, %": round(err, 4),
                     "sigma": round(sigma, 6)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))



# МНОГОКРАТНАЯ ВЫБОРКА ПО ЗНАЧИМОСТИ (MIS)
def mis_estimator(f, samplers, pdfs, N, power=1):
    """
    power=1  -> balance heuristic  (средняя плотность)
    power=2  -> power heuristic    (средний квадрат)
    """
    k = len(samplers)
    n_each = N // k
    I_parts = []
    all_samples_for_sigma = []

    for i, (sampler_i, pdf_i) in enumerate(zip(samplers, pdfs)):
        X_i = sampler_i(n_each)
        denom = np.zeros(n_each)
        for pdf_j in pdfs:
            denom += pdf_j(X_i) ** power
        numer = pdf_i(X_i) ** power
        w_i = numer / denom
        contrib = w_i * f(X_i) / pdf_i(X_i)
        I_parts.append(np.mean(contrib))
        all_samples_for_sigma.append(contrib)

    I_est = np.mean(I_parts)
    var_est = 0.0
    for contrib_arr in all_samples_for_sigma:
        var_est += np.var(contrib_arr, ddof=1) / n_each
    sigma = np.sqrt(var_est / (k ** 2))

    return I_est, sigma


print("\n=== 4. Многократная выборка по значимости (MIS) ===")
samplers_mis = [sample_p2, sample_p3]
pdfs_mis     = [p2,        p3]

for power, label in [(1, "balance (средняя плотность)"),
                     (2, "power   (средний квадрат)")]:
    print(f"\n  Вариант: {label}")
    rows = []
    for N in N_list:
        I_est, sigma = mis_estimator(f, samplers_mis, pdfs_mis, N, power=power)
        err = relative_error(I_est, I_true)
        rows.append({"N": N,
                     "I_true": round(I_true, 6),
                     "I_est": round(I_est, 6),
                     "Погрешность, %": round(err, 4),
                     "sigma": round(sigma, 6)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))



# МЕТОД МОНТЕ-КАРЛО С РУССКОЙ РУЛЕТКОЙ
def mc_russian_roulette(f, a, b, N, q_frac):

    X = np.random.uniform(a, b, N)
    U = np.random.uniform(0, 1, N)

    survive = U < q_frac # маска выживших лучей
    contrib = np.where(survive, (b - a) * f(X) / q_frac, 0.0)

    I_est = np.mean(contrib)
    sigma = std_error(contrib)
    return I_est, sigma


print("\n=== 5. Русская рулетка ===")
q_values = [0.5, 0.75, 0.95]
for q in q_values:
    print(f"\n  Вероятность выживания q = {q}")
    rows = []
    for N in N_list:
        I_est, sigma = mc_russian_roulette(f, a, b, N, q)
        err = relative_error(I_est, I_true)
        rows.append({"N": N,
                     "I_true": round(I_true, 6),
                     "I_est": round(I_est, 6),
                     "Погрешность, %": round(err, 4),
                     "sigma": round(sigma, 6)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))