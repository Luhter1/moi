import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Light:
    """Источник света"""
    I0: np.ndarray
    O: np.ndarray
    P_L: np.ndarray

@dataclass
class Surface:
    """Свойства поверхности"""
    K: np.ndarray
    kd: float # Коэффициент диффузного отражения
    ks: float # Коэффициент зеркального отражения
    ke: float # Коэффициент ширины блика

@dataclass
class Triangle:
    """Треугольник, заданный тремя вершинами"""
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    """Нормализация вектора"""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def compute_normal(tri: Triangle) -> np.ndarray:
    """Единичный вектор нормали к плоскости треугольника"""
    v1 = tri.P1 - tri.P0
    v2 = tri.P2 - tri.P0
    cross = np.cross(v2, v1)
    return normalize(cross)


def local_to_global(tri: Triangle, x: float, y: float) -> np.ndarray:
    """
    Перевод локальных координат (x, y) в глобальные
    """
    e1 = normalize(tri.P1 - tri.P0)
    e2 = normalize(tri.P2 - tri.P0)
    return tri.P0 + e1 * x + e2 * y


def compute_illuminance(P_T: np.ndarray, light: Light, N: np.ndarray) -> np.ndarray:
    """
    Вычисление освещённости E(RGB, P_T) от одного источника
    """
    s = P_T - light.P_L  # вектор от источника к точке
    R2 = np.dot(s, s)
    R = np.sqrt(R2)
    
    if R < 1e-12:
        return np.zeros(3)
    
    s_norm = s / R

    cos_alpha = np.dot(s_norm, N)

    cos_theta = np.dot(s_norm, light.O)
    
    if cos_alpha <= 0 or cos_theta <= 0:
        return np.zeros(3)
    
    I_s = light.I0 * cos_theta
    
    E = I_s * cos_alpha / R2
    
    return E


def compute_brdf(N: np.ndarray, v: np.ndarray, s: np.ndarray, 
                 surface: Surface) -> np.ndarray:
    """
    Вычисление BRDF
    """
    s_dir = normalize(s)  # нормализованное направление на источник
    v_norm = normalize(v)
    
    h_unnorm = v_norm + s_dir
    h = normalize(h_unnorm)
    
    h_dot_N = np.dot(h, N)
    h_dot_N = max(h_dot_N, 0.0)
    
    brdf_value = surface.kd + surface.ks * (h_dot_N ** surface.ke)
    
    return surface.K * brdf_value


def compute_luminance(P_T: np.ndarray, lights: List[Light], 
                      tri: Triangle, surface: Surface, 
                      V: np.ndarray) -> np.ndarray:
    """
    Вычисление яркости точки
    
    @param P_T: глобальные координаты точки
    @param lights: список источников света
    @param tri: треугольник
    @param surface: свойства поверхности
    @param V: положение наблюдателя
    @return: яркость L(RGB) в точке
    """
    N = compute_normal(tri)
    v = V - P_T  # вектор от точки к наблюдателю
    
    if np.dot(normalize(v), N) > 0:
        return np.zeros(3)
    
    L = np.zeros(3)
    
    for light in lights:
        E = compute_illuminance(P_T, light, N)
        
        s = P_T - light.P_L  # вектор от источника к точке
        brdf = compute_brdf(N, v, s, surface)
        
        L += E * brdf
    
    L = L / np.pi
    
    return L

def is_inside_triangle(tri, x, y):
    """Проверка, что точка с локальными координатами (x,y) внутри треугольника"""
    P_T = local_to_global(tri, x, y)
    
    # Вычисляем барицентрические координаты
    v0 = tri.P2 - tri.P0
    v1 = tri.P1 - tri.P0
    v2 = P_T - tri.P0
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def main():
    ###################### ВХОДНЫЕ ДАННЫЕ
    
    # Сила излучения источников
    I01_RGB = np.array([100.0, 80.0, 60.0])
    I02_RGB = np.array([50.0, 90.0, 70.0])
    
    # Направление излучения
    O1 = normalize(np.array([1.0, -1.0, -1.0]))
    O2 = normalize(np.array([0.0, 0.0, -1.0]))
    
    # Положения источников света
    P_L1 = np.array([0.0, 5.0, 5.0])
    P_L2 = np.array([-3.0, 4.0, 3.0])
    
    # Вершины треугольника
    P0 = np.array([0.0, 0.0, 0.0])
    P1 = np.array([4.0, 0.0, 0.0])
    P2 = np.array([2.0, 0.0, -3.0])
    
    # Локальные координаты точек
    x_values = [0.5, 1.0, 1.5, 1.0, 1.5]
    y_values = [0.5, 1.0, 1.5, 2.0, 1.5]
    
    # Положение наблюдателя
    V = np.array([2.0, 5.0, 0.0])
    
    # Цвет поверхности
    K_RGB = np.array([0.8, 0.6, 0.4])

    kd = 0.7 # Коэффициент диффузного отражения
    ks = 0.3 # Коэффициент зеркального отражения
    ke = 10.0 # Коэффициент ширины блика
    
    ###################### СОЗДАНИЕ ОБЪЕКТОВ
    
    light1 = Light(I0=I01_RGB, O=O1, P_L=P_L1)
    light2 = Light(I0=I02_RGB, O=O2, P_L=P_L2)
    lights = [light1, light2]
    
    tri = Triangle(P0=P0, P1=P1, P2=P2)
    surface = Surface(K=K_RGB, kd=kd, ks=ks, ke=ke)
    
    N = compute_normal(tri)

    ###################### ПРОВЕРКА НА ПРИНАДЛЕЖНОСТЬ ТОЧЕК ТРЕУГОЛЬНИКУ
    for y in y_values:
        for x in x_values:
            if not is_inside_triangle(tri, x, y):
                raise Exception(f"Точка ({x}, {y}) не внутри треуголника")
    
    ###################### ВЫВОД ВХОДНЫХ ДАННЫХ
    
    print("=" * 70)
    print("ВХОДНЫЕ ДАННЫЕ")
    print("=" * 70)
    print(f"1. I01(RGB) = ({I01_RGB[0]}, {I01_RGB[1]}, {I01_RGB[2]})")
    print(f"   I02(RGB) = ({I02_RGB[0]}, {I02_RGB[1]}, {I02_RGB[2]})")
    print(f"2. O1 = ({O1[0]:.4f}, {O1[1]:.4f}, {O1[2]:.4f})")
    print(f"   O2 = ({O2[0]:.4f}, {O2[1]:.4f}, {O2[2]:.4f})")
    print(f"3. P_L1 = ({P_L1[0]}, {P_L1[1]}, {P_L1[2]})")
    print(f"   P_L2 = ({P_L2[0]}, {P_L2[1]}, {P_L2[2]})")
    print(f"4. P0 = ({P0[0]}, {P0[1]}, {P0[2]})")
    print(f"5. P1 = ({P1[0]}, {P1[1]}, {P1[2]})")
    print(f"6. P2 = ({P2[0]}, {P2[1]}, {P2[2]})")
    print(f"7. Локальные координаты точек:")
    for i, (x, y) in enumerate([(x, y) for y in y_values for x in x_values][:5]):
        print(f"   x{i+1} = {x}, y{i+1} = {y}")
    print(f"8. V = ({V[0]}, {V[1]}, {V[2]})")
    print(f"9. K(RGB) = ({K_RGB[0]}, {K_RGB[1]}, {K_RGB[2]})")
    print(f"10. kd = {kd}")
    print(f"11. ks = {ks}")
    print(f"12. ke = {ke}")
    print(f"\nНормаль к плоскости: N = ({N[0]:.4f}, {N[1]:.4f}, {N[2]:.4f})")
    
    ###################### ВЫЧИСЛЕНИЯ
    
    # Таблица освещённости E1 ()
    print("\n" + "=" * 70)
    print("Освещённость E1(RGB, P_T) от источника 1")
    print("=" * 70)
    
    header = f"{'x→':>8}"
    for x in x_values:
        header += f"{'x='+str(x):>20}"
    print(header)
    print(f"{'y↓':>8}")
    
    E1_table = {}
    for y in y_values:
        row = f"{y:>8.1f}"
        for x in x_values:
            P_T = local_to_global(tri, x, y)
            E1 = compute_illuminance(P_T, light1, N)
            E1_table[(x, y)] = E1
            row += f"  ({E1[0]:5.3f},{E1[1]:5.3f},{E1[2]:5.3f})"
        print(row)
    
    # Таблица освещённости E2
    print("\n" + "=" * 70)
    print("Освещённость E2(RGB, P_T) от источника 2")
    print("=" * 70)
    
    print(header)
    print(f"{'y↓':>8}")
    
    E2_table = {}
    for y in y_values:
        row = f"{y:>8.1f}"
        for x in x_values:
            P_T = local_to_global(tri, x, y)
            E2 = compute_illuminance(P_T, light2, N)
            E2_table[(x, y)] = E2
            row += f"  ({E2[0]:5.3f},{E2[1]:5.3f},{E2[2]:5.3f})"
        print(row)
    
    # Таблица яркости L
    print("\n" + "=" * 70)
    print("Яркость L(RGB, P_T, v)")
    print("=" * 70)
    
    print(header)
    print(f"{'y↓':>8}")
    
    for y in y_values:
        row = f"{y:>8.1f}"
        for x in x_values:
            P_T = local_to_global(tri, x, y)
            L = compute_luminance(P_T, lights, tri, surface, V)
            row += f"  ({L[0]:5.3f},{L[1]:5.3f},{L[2]:5.3f})"
        print(row)

if __name__ == "__main__":
    main()