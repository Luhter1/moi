import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def verify_triangle_uniformity(P, V1, V2, V3, N_zones=10):
    """
    Проверка равномерности распределения в треугольнике.
    Разбиваем треугольник на зоны равной вероятности и сравниваем количество точек.
    
    Для метода сэмплирования с α = 1 - √r₁:
    - Функция распределения: F(α) = 2α - α²
    - Для ЗОН РАВНОЙ ВЕРОЯТНОСТИ границы: α_i = 1 - √(1 - i/N_zones)
    """
    N = len(P)
    expected_per_zone = N / N_zones
    
    # Вычисляем площадь треугольника
    def triangle_area(A, B, C):
        return 0.5 * abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]))
    
    total_area = triangle_area(V1, V2, V3)
    
    # Границы зон для равной вероятности при F(α) = 2α - α²
    # Решаем: 2α - α² = i/N_zones → α = 1 - √(1 - i/N_zones)
    zone_boundaries = [1 - np.sqrt(1 - i / N_zones) for i in range(N_zones + 1)]
    
    # Вычисляем барицентрические координаты α для всех точек
    # α = площадь(P,V2,V3) / площадь(V1,V2,V3)
    alphas = []
    for point in P:
        sub_area = triangle_area(point, V2, V3)
        α = sub_area / total_area
        alphas.append(α)
    alphas = np.array(alphas)
    
    counts = []
    for i in range(N_zones):
        α_low = zone_boundaries[i]
        α_high = zone_boundaries[i + 1]
        
        in_zone = np.sum((alphas >= α_low) & (alphas < α_high))
        counts.append(in_zone)
    
    # Визуализация проверки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    zone_centers = [(zone_boundaries[i] + zone_boundaries[i+1]) / 2 for i in range(N_zones)]
    x = np.arange(N_zones)
    
    # Гистограмма распределения по зонам
    ax1.bar(x, counts, width=0.8, alpha=0.7, label='Фактическое', edgecolor='black')
    ax1.axhline(expected_per_zone, color='red', linestyle='--', 
                label=f'Ожидаемое: {expected_per_zone:.0f}')
    ax1.set_xlabel('Зона (по барицентрической координате α)')
    ax1.set_ylabel('Количество точек')
    ax1.set_title('Проверка равномерности в треугольнике')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in x])
    
    # Отклонение в процентах
    deviations = [(c - expected_per_zone) / expected_per_zone * 100 for c in counts]
    ax2.bar(x, deviations, width=0.8, alpha=0.7, color='orange', edgecolor='black')
    ax2.axhline(0, color='red', linestyle='-', linewidth=2)
    ax2.set_xlabel('Зона')
    ax2.set_ylabel('Отклонение (%)')
    ax2.set_title('Отклонение от ожидаемого количества')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i+1}' for i in x])
    
    plt.tight_layout()
    plt.savefig('triangle_uniformity.png', dpi=150)
    plt.close()
    
    # Статистика
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    max_deviation = max(abs(d) for d in deviations)
    
    print(f"\n=== Проверка равномерности: ТРЕУГОЛЬНИК ===")
    print(f"Всего точек: {N}")
    print(f"Зон: {N_zones}")
    print(f"Ожидаемо в зоне: {expected_per_zone:.0f}")
    print(f"Среднее в зоне: {mean_count:.1f} ± {std_count:.1f}")
    print(f"Макс. отклонение: {max_deviation:.2f}%")
    print(f"Стандартное отклонение: {std_count:.1f} ({std_count/expected_per_zone*100:.2f}% от ожидаемого)")
    
    return counts, deviations


def plot_triangle(P, V1, V2, V3, filename="triangle.png"):
    """Визуализация точек в треугольнике"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Рисуем границы треугольника
    triangle = np.array([V1, V2, V3, V1])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, label='Границы')
    
    # Рисуем точки
    ax.scatter(P[:, 0], P[:, 1], s=0.5, alpha=0.5, c='blue', label='Точки')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Распределение точек в треугольнике')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Изображение сохранено: {filename}")

def sample_triangle(V1, V2, V3, N=100000):
    r1 = np.random.rand(N)
    r2 = np.random.rand(N)

    sqrt_r1 = np.sqrt(r1)

    P = (1 - sqrt_r1)[:, None] * V1 + \
        (sqrt_r1 * (1 - r2))[:, None] * V2 + \
        (sqrt_r1 * r2)[:, None] * V3
    return P

def verify_disk_uniformity(P, C, Rc, N_zones=10):
    """
    Проверка равномерности распределения в диске.
    Разбиваем диск на концентрические кольца и сравниваем количество точек.
    Для равномерного распределения количество точек пропорционально площади кольца.
    """
    N = len(P)
    
    # Вычисляем расстояния от центра для каждой точки
    distances = np.sqrt(np.sum((P[:, :2] - C[:2])**2, axis=1))
    
    # Разбиваем диск на кольца равной площади
    # Площадь круга радиуса r: A = πr²
    # Для равных площадей колец: r_i = Rc * √(i/N_zones)
    zone_boundaries = [Rc * np.sqrt(i / N_zones) for i in range(N_zones + 1)]
    
    counts = []
    expected_counts = []
    
    for i in range(N_zones):
        r_inner = zone_boundaries[i]
        r_outer = zone_boundaries[i + 1]
        
        # Площадь кольца: π(r_outer² - r_inner²)
        ring_area = np.pi * (r_outer**2 - r_inner**2)
        total_area = np.pi * Rc**2
        
        # Ожидаемое количество точек пропорционально площади
        expected = N * ring_area / total_area
        expected_counts.append(expected)
        
        # Подсчёт точек в кольце
        in_ring = np.sum((distances >= r_inner) & (distances < r_outer))
        counts.append(in_ring)
    
    # Визуализация проверки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    zone_centers = [(zone_boundaries[i] + zone_boundaries[i+1]) / 2 for i in range(N_zones)]
    
    # Гистограмма распределения по зонам
    width = (zone_boundaries[1] - zone_boundaries[0]) * 0.8
    x = np.arange(N_zones)
    
    ax1.bar(x, counts, width=0.8, alpha=0.7, label='Фактическое', edgecolor='black')
    ax1.plot(x, expected_counts, 'r--', marker='o', label=f'Ожидаемое')
    ax1.set_xlabel('Зона (кольцо)')
    ax1.set_ylabel('Количество точек')
    ax1.set_title('Проверка равномерности в диске')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in x])
    
    # Отклонение в процентах
    deviations = [(counts[i] - expected_counts[i]) / expected_counts[i] * 100 
                  for i in range(N_zones)]
    ax2.bar(x, deviations, width=0.8, alpha=0.7, color='orange', edgecolor='black')
    ax2.axhline(0, color='red', linestyle='-', linewidth=2)
    ax2.set_xlabel('Зона (кольцо)')
    ax2.set_ylabel('Отклонение (%)')
    ax2.set_title('Отклонение от ожидаемого количества')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i+1}' for i in x])
    
    plt.tight_layout()
    plt.savefig('disk_uniformity.png', dpi=150)
    plt.close()
    
    # Статистика
    mean_deviation = np.mean([abs(d) for d in deviations])
    max_deviation = max(abs(d) for d in deviations)
    std_count = np.std(counts)
    
    print(f"\n=== Проверка равномерности: ДИСК ===")
    print(f"Всего точек: {N}")
    print(f"Зон (колец): {N_zones}")
    print(f"Среднее отклонение: {mean_deviation:.2f}%")
    print(f"Макс. отклонение: {max_deviation:.2f}%")
    print(f"Стандартное отклонение: {std_count:.1f}")
    
    return counts, deviations


def plot_disk(P, C, N_vec, Rc, filename="disk.png"):
    """Визуализация точек в диске"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Рисуем границу диска (круг)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = Rc * np.cos(theta)
    circle_y = Rc * np.sin(theta)
    ax.plot(circle_x, circle_y, 'k-', linewidth=2, label='Граница')
    
    # Рисуем точки
    ax.scatter(P[:, 0], P[:, 1], s=0.5, alpha=0.5, c='green', label='Точки')
    
    # Центр
    ax.scatter(C[0], C[1], c='red', s=50, marker='x', label='Центр')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Распределение точек в диске')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Изображение сохранено: {filename}")

def sample_disk(C, N_vec, Rc, N=100000):
    u1 = np.random.rand(N)
    u2 = np.random.rand(N)

    r = Rc * np.sqrt(u1)
    phi = 2 * np.pi * u2

    # построение базиса
    T1 = np.cross(N_vec, [1,0,0])
    # если N параллелен (1,0,0), то норма нулевая, на нее делить нельзя
    if np.linalg.norm(T1) < 1e-6:
        T1 = np.cross(N_vec, [0,1,0])
    T1 /= np.linalg.norm(T1)
    T2 = np.cross(N_vec, T1)

    P = C + (r * np.cos(phi))[:,None]*T1 + \
            (r * np.sin(phi))[:,None]*T2
    return P

def verify_sphere_uniformity(P, N_zones=10):
    """
    Проверка равномерности распределения на сфере.
    Разбиваем сферу на зоны по широте (z-координате) и сравниваем количество точек.
    Для равномерного распределения количество точек пропорционально площади зоны.
    """
    N = len(P)
    
    # Вычисляем z-координаты (косинус полярного угла)
    z = P[:, 2]
    
    # Разбиваем сферу на зоны равной площади по z
    # Площадь сферической зоны между z1 и z2: A = 2πR²(z2 - z1)
    # Для единичной сферы и равных площадей: z_i = -1 + 2*i/N_zones
    zone_boundaries = np.linspace(-1, 1, N_zones + 1)
    
    counts = []
    expected_counts = []
    
    for i in range(N_zones):
        z_low = zone_boundaries[i]
        z_high = zone_boundaries[i + 1]
        
        # Площадь зоны пропорциональна разнице z (для единичной сферы)
        zone_area = 2 * np.pi * (z_high - z_low)  # R = 1
        total_area = 4 * np.pi  # Площадь всей сферы
        
        # Ожидаемое количество точек пропорционально площади
        expected = N * zone_area / total_area
        expected_counts.append(expected)
        
        # Подсчёт точек в зоне
        in_zone = np.sum((z >= z_low) & (z < z_high))
        counts.append(in_zone)
    
    # Визуализация проверки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    zone_centers = (zone_boundaries[:-1] + zone_boundaries[1:]) / 2
    x = np.arange(N_zones)
    
    # Гистограмма распределения по зонам
    ax1.bar(x, counts, width=0.8, alpha=0.7, label='Фактическое', edgecolor='black')
    ax1.plot(x, expected_counts, 'r--', marker='o', label='Ожидаемое')
    ax1.set_xlabel('Зона (по z-координате)')
    ax1.set_ylabel('Количество точек')
    ax1.set_title('Проверка равномерности на сфере')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in x])
    
    # Отклонение в процентах
    deviations = [(counts[i] - expected_counts[i]) / expected_counts[i] * 100 
                  for i in range(N_zones)]
    ax2.bar(x, deviations, width=0.8, alpha=0.7, color='orange', edgecolor='black')
    ax2.axhline(0, color='red', linestyle='-', linewidth=2)
    ax2.set_xlabel('Зона (по z-координате)')
    ax2.set_ylabel('Отклонение (%)')
    ax2.set_title('Отклонение от ожидаемого количества')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i+1}' for i in x])
    
    plt.tight_layout()
    plt.savefig('sphere_uniformity.png', dpi=150)
    plt.close()
    
    # Статистика
    mean_deviation = np.mean([abs(d) for d in deviations])
    max_deviation = max(abs(d) for d in deviations)
    std_count = np.std(counts)
    
    print(f"\n=== Проверка равномерности: СФЕРА ===")
    print(f"Всего точек: {N}")
    print(f"Зон (по широте): {N_zones}")
    print(f"Среднее отклонение: {mean_deviation:.2f}%")
    print(f"Макс. отклонение: {max_deviation:.2f}%")
    print(f"Стандартное отклонение: {std_count:.1f}")
    
    return counts, deviations


def plot_sphere(P, filename="sphere.png"):
    """Визуализация точек на сфере"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Рисуем точки
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.5, alpha=0.5, c='red', label='Точки')
    
    # Рисуем экватор для наглядности
    theta = np.linspace(0, 2*np.pi, 100)
    equator_x = np.cos(theta)
    equator_y = np.sin(theta)
    equator_z = np.zeros_like(theta)
    ax.plot(equator_x, equator_y, equator_z, 'k-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Распределение точек на сфере')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Изображение сохранено: {filename}")

def sample_sphere(N=100000):
    u1 = np.random.rand(N)
    u2 = np.random.rand(N)

    phi = 2*np.pi*u1
    z = 1 - 2*u2
    r = np.sqrt(1 - z**2)

    x = r*np.cos(phi)
    y = r*np.sin(phi)

    return np.stack([x,y,z], axis=1)

def verify_cosine_hemisphere_uniformity(P, N_zones=10):
    """
    Проверка косинусного распределения на полусфере.
    Разбиваем полусферу на зоны по z-координате и сравниваем количество точек.
    Для косинусного распределения плотность пропорциональна z (cos θ).
    """
    N = len(P)
    
    # Вычисляем z-координаты
    z = P[:, 2]
    
    # Разбиваем полусферу на зоны по z от 0 до 1
    zone_boundaries = np.linspace(0, 1, N_zones + 1)
    
    counts = []
    expected_counts = []
    
    # Для косинусного распределения f(z) = 2z (нормированная плотность)
    # Вероятность попасть в зону [z_low, z_high]:
    # P = ∫ 2z dz = z² | от z_low до z_high = z_high² - z_low²
    
    for i in range(N_zones):
        z_low = zone_boundaries[i]
        z_high = zone_boundaries[i + 1]
        
        # Вероятность для косинусного распределения
        prob = z_high**2 - z_low**2
        expected = N * prob
        expected_counts.append(expected)
        
        # Подсчёт точек в зоне
        in_zone = np.sum((z >= z_low) & (z < z_high))
        counts.append(in_zone)
    
    # Визуализация проверки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    zone_centers = (zone_boundaries[:-1] + zone_boundaries[1:]) / 2
    x = np.arange(N_zones)
    
    # Гистограмма распределения по зонам
    ax1.bar(x, counts, width=0.8, alpha=0.7, label='Фактическое', edgecolor='black')
    ax1.plot(x, expected_counts, 'r--', marker='o', label='Ожидаемое (cos)')
    ax1.set_xlabel('Зона (по z-координате)')
    ax1.set_ylabel('Количество точек')
    ax1.set_title('Проверка косинусного распределения на полусфере')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in x])
    
    # Отклонение в процентах
    deviations = [(counts[i] - expected_counts[i]) / expected_counts[i] * 100 
                  for i in range(N_zones) if expected_counts[i] > 0]
    ax2.bar(x[:len(deviations)], deviations, width=0.8, alpha=0.7, 
            color='orange', edgecolor='black')
    ax2.axhline(0, color='red', linestyle='-', linewidth=2)
    ax2.set_xlabel('Зона (по z-координате)')
    ax2.set_ylabel('Отклонение (%)')
    ax2.set_title('Отклонение от ожидаемого количества')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i+1}' for i in x])
    
    plt.tight_layout()
    plt.savefig('cosine_hemisphere_uniformity.png', dpi=150)
    plt.close()
    
    # Статистика
    mean_deviation = np.mean([abs(d) for d in deviations]) if deviations else 0
    max_deviation = max(abs(d) for d in deviations) if deviations else 0
    std_count = np.std(counts)
    
    print(f"\n=== Проверка косинусного распределения: ПОЛУСФЕРА ===")
    print(f"Всего точек: {N}")
    print(f"Зон (по z): {N_zones}")
    print(f"Среднее отклонение: {mean_deviation:.2f}%")
    print(f"Макс. отклонение: {max_deviation:.2f}%")
    print(f"Стандартное отклонение: {std_count:.1f}")
    
    return counts, deviations


def plot_cosine_hemisphere(P, filename="cosine_hemisphere.png"):
    """Визуализация точек на косинусной полусфере"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Рисуем точки
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.5, alpha=0.5, c='purple', label='Точки')
    
    # Рисуем основание полусферы (круг в плоскости XY)
    theta = np.linspace(0, 2*np.pi, 100)
    base_x = np.cos(theta)
    base_y = np.sin(theta)
    base_z = np.zeros_like(theta)
    ax.plot(base_x, base_y, base_z, 'k-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Распределение точек на косинусной полусфере')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Изображение сохранено: {filename}")

def sample_cosine_hemisphere(N=100000):
    u1 = np.random.rand(N)
    u2 = np.random.rand(N)

    r = np.sqrt(u1)
    phi = 2*np.pi*u2

    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z = np.sqrt(1 - r**2)

    return np.stack([x,y,z], axis=1)

if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ТОЧЕК И ПРОВЕРКА РАВНОМЕРНОСТИ")
    print("=" * 60)
    
    # 1. Треугольник
    V1 = np.array([0, 0])
    V2 = np.array([1, 0])
    V3 = np.array([0.5, 0.866])
    P_triangle = sample_triangle(V1, V2, V3)
    plot_triangle(P_triangle, V1, V2, V3)
    verify_triangle_uniformity(P_triangle, V1, V2, V3)
    
    # 2. Диск
    C = np.array([0.0, 0.0, 0.0])
    N_vec = np.array([0.0, 0.0, 1.0])
    Rc = 1.0
    P_disk = sample_disk(C, N_vec, Rc)
    plot_disk(P_disk, C, N_vec, Rc)
    verify_disk_uniformity(P_disk, C, Rc)
    
    # 3. Сфера
    P_sphere = sample_sphere()
    plot_sphere(P_sphere)
    verify_sphere_uniformity(P_sphere)
    
    # 4. Косинусная полусфера
    P_hemi = sample_cosine_hemisphere()
    plot_cosine_hemisphere(P_hemi)
    verify_cosine_hemisphere_uniformity(P_hemi)
    
    print("\n" + "=" * 60)
    print("=== Все изображения и проверки созданы! ===")
    print("=" * 60)
    print("\nСозданные файлы:")
    print("  Визуализации:")
    print("    - triangle.png")
    print("    - disk.png")
    print("    - sphere.png")
    print("    - cosine_hemisphere.png")
    print("  Проверки равномерности:")
    print("    - triangle_uniformity.png")
    print("    - disk_uniformity.png")
    print("    - sphere_uniformity.png")
    print("    - cosine_hemisphere_uniformity.png")