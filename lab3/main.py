import numpy as np
import os
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

EPS = 1e-6
INF = 1e30

# Вспомогательные математические функции
def luminance(c: np.ndarray) -> float:
    """
    Воспринимаемая яркость цвета, перевод линейного rgb в яркость
    RGB: 
    - Зелёный = 72%
    - Красный = 21%
    - Синий = 7%
    """    
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Зеркальное отражение вектора d от нормали n"""
    return d - 2.0 * np.dot(d, n) * n

def cosine_sample_hemisphere(normal: np.ndarray) -> np.ndarray:
    """
    Генерирует случайное направление отражения для диффузных поверхностей согласно закону Ламберта

    Где используется:
        Луч попал на диффузную поверхность
        Выбираем новое направление для продолжения пути
        Направление выбирается с учётом закона Ламберта
        Больше лучей в направлениях с большим вкладом
        Меньше шума при том же количестве лучей

    return:
        Функция возвращает вектор направления, распределённый по косинусному закону:
        Больше точек вблизи нормали
        Меньше точек у горизонта
    """
    u1, u2 = np.random.random(), np.random.random()
    # Метод Мальли
    r = np.sqrt(u1)
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(max(0.0, 1.0 - u1))

    # Строим ОНБ вокруг normal
    up = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = normalize(np.cross(up, normal))
    b = np.cross(normal, t)
    return x * t + y * b + z * normal


# Структуры данных сцены
@dataclass
class Material:
    """
    Материал поверхности
    
    эмиссия — способность поверхности самой излучать свет, а не только отражать падающий свет

    """
    diffuse:  np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))
    specular: np.ndarray = field(default_factory=lambda: np.zeros(3))
    emission: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        # Гарантируем сохранение энергии: kd + ks <= 1 покомпонентно
        total = self.diffuse + self.specular
        mask = total > 1.0
        if np.any(mask):
            scale = np.where(mask, 1.0 / total, 1.0)
            self.diffuse  = self.diffuse  * scale
            self.specular = self.specular * scale

    @property
    def is_emitter(self) -> bool:
        return np.any(self.emission > 0.0)


@dataclass
class Triangle:
    """
    Один треугольник сетки

    - v0, v1, v2 — координаты вершин
    - e1, e2 — рёбра треугольника
    - normal — нормаль треугольника
    - area — площадь треугольника
    """
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material: Material

    def __post_init__(self):
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        n  = np.cross(self.e1, self.e2)
        self.area   = 0.5 * np.linalg.norm(n)
        self.normal = normalize(n)

    def sample_point(self) -> np.ndarray:
        """Случайная точка на треугольнике (равномерно)."""
        u1, u2 = np.random.random(), np.random.random()
        if u1 + u2 > 1.0:
            u1, u2 = 1.0 - u1, 1.0 - u2
        return self.v0 + u1 * (self.v1 - self.v0) + u2 * (self.v2 - self.v0)

    def intersect(self, ray_orig: np.ndarray, ray_dir: np.ndarray
                  ) -> Optional[float]:
        """
        Алгоритм Мёллера–Трумбора. Возвращает t или None.

        Быстрый алгоритм проверки пересечения луча с треугольником без предварительного вычисления нормали.

        h = d x e2
        a = e1 * h
        if |a| < EPS: луч параллелен треугольнику (нет пересечения)

        f = 1/a
        s = o - v0
        u = f * (s * h)
        if u < 0 или u > 1: точка вне треугольника

        q = s x e1
        v = f * (d * q)
        if v < 0 или u + v > 1: точка вне треугольника

        t = f * (e2 * q)
        if t > EPS: есть пересечение на расстоянии t
        """
        h  = np.cross(ray_dir, self.e2)
        a  = np.dot(self.e1, h)
        if abs(a) < EPS:
            return None
        f  = 1.0 / a
        s  = ray_orig - self.v0
        u  = f * np.dot(s, h)
        if not (0.0 <= u <= 1.0):
            return None
        q  = np.cross(s, self.e1)
        v  = f * np.dot(ray_dir, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t  = f * np.dot(self.e2, q)
        return t if t > EPS else None


# Сцена
class Scene:
    def __init__(self):
        self.triangles: List[Triangle] = []
        self._lights:   List[Triangle] = []        # кэш источников
        self._light_powers: np.ndarray = np.array([])
        self._light_pdf_map: dict = {}
        self._accel_built = False
        # Flat arrays: interleaved xyz, shape (3*N,)
        self._v0f: np.ndarray = np.array([])
        self._e1f: np.ndarray = np.array([])
        self._e2f: np.ndarray = np.array([])
        self._n_tri: int = 0

    def add_triangle(self, tri: Triangle):
        self.triangles.append(tri)
        if tri.material.is_emitter:
            self._lights.append(tri)

    def add_triangles(self, tris: List[Triangle]):
        for t in tris:
            self.triangles.append(t)
            if t.material.is_emitter:
                self._lights.append(t)

    def _rebuild_light_cache(self):
        if not self._lights:
            return
        powers = np.array([np.sum(t.material.emission) * t.area
                           for t in self._lights])
        self._light_powers = powers / powers.sum()
        self._light_pdf_map = {id(t): i for i, t in enumerate(self._lights)}

    def build_accel(self):
        """Создать flat numpy-массивы для векторизованных пересечений."""
        self._rebuild_light_cache()
        N = len(self.triangles)
        self._n_tri = N
        if N == 0:
            return
        self._v0f = np.array([t.v0 for t in self.triangles]).ravel().astype(np.float64)
        self._e1f = np.array([t.e1 for t in self.triangles]).ravel().astype(np.float64)
        self._e2f = np.array([t.e2 for t in self.triangles]).ravel().astype(np.float64)
        self._accel_built = True

    # ── Пересечение (векторизованное, flat arrays) ──
    def intersect(self, orig: np.ndarray, direction: np.ndarray
                  ) -> Tuple[Optional[Triangle], float]:
        """Ближайшее пересечение луча со сценой."""
        d = direction
        v0f = self._v0f; e1f = self._e1f; e2f = self._e2f

        # h = cross(d, e2)
        hx = d[1]*e2f[2::3] - d[2]*e2f[1::3]
        hy = d[2]*e2f[0::3] - d[0]*e2f[2::3]
        hz = d[0]*e2f[1::3] - d[1]*e2f[0::3]

        # a = dot(e1, h)
        a = e1f[0::3]*hx + e1f[1::3]*hy + e1f[2::3]*hz

        valid = np.abs(a) > EPS
        if not np.any(valid):
            return None, INF

        a_safe = np.where(valid, a, 1.0)
        f = 1.0 / a_safe
        f[~valid] = 0.0

        # s = orig - v0
        sx = orig[0] - v0f[0::3]
        sy = orig[1] - v0f[1::3]
        sz = orig[2] - v0f[2::3]

        # u = f * dot(s, h)
        u = f * (sx*hx + sy*hy + sz*hz)
        valid &= (u >= 0.0) & (u <= 1.0)

        # q = cross(s, e1)
        qx = sy*e1f[2::3] - sz*e1f[1::3]
        qy = sz*e1f[0::3] - sx*e1f[2::3]
        qz = sx*e1f[1::3] - sy*e1f[0::3]

        # v = f * dot(d, q)
        v = f * (d[0]*qx + d[1]*qy + d[2]*qz)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        # t = f * dot(e2, q)
        t = f * (e2f[0::3]*qx + e2f[1::3]*qy + e2f[2::3]*qz)
        valid &= (t > EPS)

        if not np.any(valid):
            return None, INF

        t[~valid] = INF
        idx = int(np.argmin(t))
        return self.triangles[idx], t[idx]

    def is_occluded(self, orig: np.ndarray, direction: np.ndarray,
                    max_t: float) -> bool:
        """Проверка видимости (тень)."""
        d = direction
        v0f = self._v0f; e1f = self._e1f; e2f = self._e2f

        hx = d[1]*e2f[2::3] - d[2]*e2f[1::3]
        hy = d[2]*e2f[0::3] - d[0]*e2f[2::3]
        hz = d[0]*e2f[1::3] - d[1]*e2f[0::3]

        a = e1f[0::3]*hx + e1f[1::3]*hy + e1f[2::3]*hz
        valid = np.abs(a) > EPS
        if not np.any(valid):
            return False

        a_safe = np.where(valid, a, 1.0)
        f = 1.0 / a_safe
        f[~valid] = 0.0

        sx = orig[0] - v0f[0::3]
        sy = orig[1] - v0f[1::3]
        sz = orig[2] - v0f[2::3]

        u = f * (sx*hx + sy*hy + sz*hz)
        valid &= (u >= 0.0) & (u <= 1.0)

        qx = sy*e1f[2::3] - sz*e1f[1::3]
        qy = sz*e1f[0::3] - sx*e1f[2::3]
        qz = sx*e1f[1::3] - sy*e1f[0::3]

        v = f * (d[0]*qx + d[1]*qy + d[2]*qz)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        t = f * (e2f[0::3]*qx + e2f[1::3]*qy + e2f[2::3]*qz)
        valid &= (t > EPS) & (t < max_t - EPS)

        return bool(np.any(valid))

    # Выборка источника
    def sample_light(self) -> Optional[Triangle]:
        if not self._lights:
            return None
        idx = np.random.choice(len(self._lights), p=self._light_powers)
        return self._lights[idx]

    def light_pdf(self, light: Triangle) -> float:
        """Полный pdf выборки точки на источнике: p_select / area."""
        idx = self._light_pdf_map.get(id(light))
        if idx is None:
            return 0.0
        return self._light_powers[idx] / light.area


# ──────────────────────────────────────────────
# Камера
# ──────────────────────────────────────────────

class Camera:
    '''
    - position — положение камеры
    - look_at — точка, куда смотрит камера
    - up — вектор "верха"
    - fov_deg — поле зрения в градусах
    - width, height — разрешение изображения
    '''
    def __init__(self, position: np.ndarray, look_at: np.ndarray,
                 up: np.ndarray, fov_deg: float,
                 width: int, height: int):
        self.position = position
        self.width    = width
        self.height   = height

        fwd = normalize(look_at - position)     # Ось Z (направление)
        rgt = normalize(np.cross(fwd, up))      # Ось X (вправо)
        u   = np.cross(rgt, fwd)                # Ось Y (вверх)

        half_h = np.tan(np.radians(fov_deg / 2.0))
        half_w = half_h * width / height

        self.lower_left = fwd - half_w * rgt - half_h * u # Нижний левый угол экрана
        self.horiz      = 2.0 * half_w * rgt
        self.vert       = 2.0 * half_h * u
        self.right      = rgt
        self.up         = u

    def get_ray(self, px: int, py: int
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (origin, direction) с антиалиасингом."""
        # Случайное смещение внутри пикселя, так как дискретизация пикселей вызывает ступенчатые края
        sx = (px + np.random.random()) / self.width
        sy = (py + np.random.random()) / self.height
        direction = normalize(self.lower_left + sx * self.horiz + sy * self.vert) # Направление луча для пикселя
        return self.position.copy(), direction


# ──────────────────────────────────────────────
# Трассировщик путей
# ──────────────────────────────────────────────

class PathTracer:
    def __init__(self, scene: Scene, camera: Camera,
                 max_depth: int = 8,
                 rr_start_depth: int = 3):
        self.scene         = scene
        self.camera        = camera
        self.max_depth     = max_depth
        self.rr_start_depth = rr_start_depth

    # ── Одна выборка пути ──────────────────────
    def trace(self, orig: np.ndarray, direction: np.ndarray) -> np.ndarray:
        color      = np.zeros(3)
        throughput = np.ones(3)

        for depth in range(self.max_depth):

            tri, t = self.scene.intersect(orig, direction)

            if tri is None:
                break

            hit_point = orig + t * direction
            mat       = tri.material
            normal    = tri.normal

            # Нормаль смотрит навстречу лучу
            if np.dot(normal, direction) > 0:
                normal = -normal

            # Эмиссия (источник света)
            # путь завершается для упрощения
            if mat.is_emitter:
                color += throughput * mat.emission
                break

            # Русская рулетка
            total_refl = mat.diffuse + mat.specular
            p_continue = min(max(total_refl[0], total_refl[1], total_refl[2]), 0.95)
            if depth >= self.rr_start_depth:
                if np.random.random() > p_continue:
                    break
                throughput = throughput / p_continue

            # Выбор типа отражения
            diff_weight = max(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2])
            spec_weight = max(mat.specular[0], mat.specular[1], mat.specular[2])
            total_weight = diff_weight + spec_weight

            if total_weight < EPS:
                break

            p_diff = diff_weight / total_weight

            if np.random.random() < p_diff:
                # Диффузное отражение
                # NEE: прямое освещение
                color += throughput * self._direct_light(hit_point, normal, mat)
                # Продолжаем путь
                new_dir = cosine_sample_hemisphere(normal)
                throughput = throughput * mat.diffuse
            else:
                # Зеркальное отражение
                new_dir = reflect(direction, normal)
                throughput = throughput * mat.specular

            orig      = hit_point + normal * EPS
            direction = new_dir

        return color

    # Прямое освещение
    # Вместо того чтобы надеяться на случайное попадание в источник, 
    # мы выпускаем дополнительный луч прямо к случайной точке на источнике света
    def _direct_light(self, point: np.ndarray, normal: np.ndarray,
                      mat: Material) -> np.ndarray:
        """Выборка прямого освещения через один случайный источник."""
        light = self.scene.sample_light()
        if light is None:
            return np.zeros(3)

        # выбираем случайную точку на источнике
        light_point  = light.sample_point()
        to_light     = light_point - point
        dist         = np.linalg.norm(to_light)
        if dist < EPS:
            return np.zeros(3)
        to_light_n   = to_light / dist

        cos_surf  = np.dot(normal, to_light_n)
        cos_light = np.dot(-light.normal, to_light_n)

        if cos_surf <= 0 or cos_light <= 0:
            return np.zeros(3)

        # Проверка тени
        if self.scene.is_occluded(point + normal * EPS, to_light_n, dist):
            return np.zeros(3)

        # Полный pdf выборки: p_select / area
        pdf_light = self.scene.light_pdf(light)
        if pdf_light < EPS:
            return np.zeros(3)

        # Геометрический терм
        geom = cos_surf * cos_light / (dist * dist)
        # Ламбертовская BRDF: kd/π
        # Вклад: Le * (kd/π) * geom / pdf_light
        # делим на pdf, чтобы компенсировать то, что яркие берем часто, а неяркие редко
        contrib = light.material.emission * (mat.diffuse / np.pi) * geom / pdf_light
        return contrib


# Глобальные переменные воркеров
_w_tracer: Optional[PathTracer] = None
_w_camera: Optional[Camera] = None


def _worker_init(tracer: PathTracer, camera: Camera):
    global _w_tracer, _w_camera
    _w_tracer = tracer
    _w_camera = camera
    # Уникальный seed для каждого воркера
    np.random.seed(os.getpid())


def _render_row(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """Рендерит одну строку изображения."""
    py, spp = args
    W = _w_camera.width
    row = np.zeros((W, 3))
    for px in range(W):
        acc = np.zeros(3)
        for _ in range(spp):
            orig, direction = _w_camera.get_ray(px, py)
            acc += _w_tracer.trace(orig, direction)
        row[px] = acc / spp
    return py, row


def render(scene: Scene, camera: Camera,
           spp: int = 64,
           max_depth: int = 8,
           gamma: float = 2.2,
           exposure: float = 1.0,
           output_path: str = "output.ppm") -> np.ndarray:
    """
    Основная функция рендера.

    Parameters
    ----------
    spp        : число лучей на пиксель
    max_depth  : максимальная глубина пути
    gamma      : параметр гамма-коррекции
    exposure   : коэффициент экспозиции (масштаб яркости перед тоном)
    output_path: путь к выходному PPM-файлу
    """
    W, H = camera.width, camera.height
    scene.build_accel()
    tracer = PathTracer(scene, camera, max_depth=max_depth)

    hdr_buffer = np.zeros((H, W, 3), dtype=np.float64)

    total_pixels = W * H
    start_time   = time.time()

    ncpus = mp.cpu_count()

    if ncpus > 1:
        print(f"  Используем {ncpus} процессов")
        with mp.Pool(ncpus, initializer=_worker_init,
                     initargs=(tracer, camera)) as pool:
            for i, (py, row) in enumerate(
                    pool.imap_unordered(_render_row,
                                        [(py, spp) for py in range(H)])):
                hdr_buffer[py] = row
                elapsed = time.time() - start_time
                pct = (i + 1) * 100 // H
                eta = elapsed / (i + 1) * (H - i - 1)
                print(f"\r  Строка {i+1:4d}/{H} ({pct:3d}%)  "
                      f"ETA {eta:6.1f} с   ", end="", flush=True)
    else:
        for py in range(H):
            for px in range(W):
                acc = np.zeros(3)
                for _ in range(spp):
                    orig, direction = camera.get_ray(px, py)
                    acc += tracer.trace(orig, direction)
                hdr_buffer[py, px] = acc / spp

            elapsed = time.time() - start_time
            done    = (py + 1) * W
            eta     = elapsed / done * (total_pixels - done) if done else 0
            print(f"\r  Строка {py+1:4d}/{H}  "
                  f"ETA {eta:6.1f} с   ", end="", flush=True)

    print(f"\nГотово за {time.time()-start_time:.1f} с")

    # Тональная компрессия
    img = hdr_buffer * exposure
    # Нормировка по средней яркости -> 0.5
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    # Среднее по ненулевым пикселям (исключаем фон)
    nonzero = lum[lum > EPS]
    if len(nonzero) > 0:
        mean_lum = np.mean(nonzero)
        img *= 0.5 / mean_lum
    np.clip(img, 0.0, 1.0, out=img)

    # Гамма-коррекция
    img_gamma = np.power(img, 1.0 / gamma)

    # Перевод в 0-255
    img_uint8 = (img_gamma * 255.0).astype(np.uint8)

    # Запись PPM
    _save_ppm(img_uint8, output_path)
    print(f"Изображение сохранено: {output_path}")

    return hdr_buffer


def _save_ppm(img: np.ndarray, path: str):
    H, W, _ = img.shape
    with open(path, "wb") as f:
        header = f"P6\n{W} {H}\n255\n"
        f.write(header.encode())
        f.write(img.tobytes())


# Построение тестовой сцены — Корнельская коробка
def build_cornell_box() -> Tuple[Scene, Camera]:
    scene = Scene()

    # Материалы
    white  = Material(diffuse=np.array([0.73, 0.73, 0.73]))
    red    = Material(diffuse=np.array([0.65, 0.05, 0.05]))
    green  = Material(diffuse=np.array([0.12, 0.45, 0.15]))
    mirror = Material(diffuse=np.zeros(3),
                      specular=np.array([0.95, 0.95, 0.95]))
    mixed  = Material(diffuse=np.array([0.5, 0.4, 0.1]),
                      specular=np.array([0.4, 0.4, 0.4]))
    light_mat = Material(emission=np.array([8.0, 8.0, 6.5]))

    def quad(v0, v1, v2, v3, mat):
        """Квад → два треугольника."""
        return [Triangle(np.array(v0), np.array(v1), np.array(v2), mat),
                Triangle(np.array(v0), np.array(v2), np.array(v3), mat)]

    # Пол
    scene.add_triangles(quad(
        [0,0,0],[1,0,0],[1,0,1],[0,0,1], white))
    # Потолок
    scene.add_triangles(quad(
        [0,1,0],[0,1,1],[1,1,1],[1,1,0], white))
    # Задняя стена
    scene.add_triangles(quad(
        [0,0,1],[1,0,1],[1,1,1],[0,1,1], white))
    # Левая стена (красная)
    scene.add_triangles(quad(
        [0,0,0],[0,0,1],[0,1,1],[0,1,0], red))
    # Правая стена (зелёная)
    scene.add_triangles(quad(
        [1,0,0],[1,1,0],[1,1,1],[1,0,1], green))

    # Источник света на потолке
    scene.add_triangles(quad(
        [0.35,0.999,0.35],[0.65,0.999,0.35],
        [0.65,0.999,0.65],[0.35,0.999,0.65], light_mat))

    # Блок с поворотом вокруг центра по оси Y
    def rotated_box(x0, y0, z0, x1, y1, z1, mat, angle_deg=0):
        cx = (x0 + x1) / 2
        cz = (z0 + z1) / 2
        a = np.radians(angle_deg)
        ca, sa = np.cos(a), np.sin(a)

        def rot(v):
            dx, dz = v[0] - cx, v[2] - cz
            return [cx + dx*ca - dz*sa, v[1], cz + dx*sa + dz*ca]

        tris = []
        tris += quad(rot([x0,y0,z0]), rot([x1,y0,z0]),
                     rot([x1,y0,z1]), rot([x0,y0,z1]), mat)  # дно
        tris += quad(rot([x0,y1,z0]), rot([x0,y1,z1]),
                     rot([x1,y1,z1]), rot([x1,y1,z0]), mat)  # крыша
        tris += quad(rot([x0,y0,z0]), rot([x0,y0,z1]),
                     rot([x0,y1,z1]), rot([x0,y1,z0]), mat)  # лево
        tris += quad(rot([x1,y0,z0]), rot([x1,y1,z0]),
                     rot([x1,y1,z1]), rot([x1,y0,z1]), mat)  # право
        tris += quad(rot([x0,y0,z0]), rot([x0,y1,z0]),
                     rot([x1,y1,z0]), rot([x1,y0,z0]), mat)  # перед
        tris += quad(rot([x0,y0,z1]), rot([x1,y0,z1]),
                     rot([x1,y1,z1]), rot([x0,y1,z1]), mat)  # зад
        return tris

    # Высокий зеркальный блок (повёрнут, чтобы отражать интерьер)
    scene.add_triangles(rotated_box(0.55, 0.0, 0.42,
                                    0.82, 0.6, 0.70, mirror, angle_deg=-30))
    # Низкий блок со смешанным материалом (повёрнут)
    scene.add_triangles(rotated_box(0.18, 0.0, 0.18,
                                    0.45, 0.3, 0.45, mixed, angle_deg=20))

    # Камера
    camera = Camera(
        position = np.array([0.5, 0.5, -1.4]),
        look_at  = np.array([0.5, 0.5, 0.5]),
        up       = np.array([0.0, 1.0, 0.0]),
        fov_deg  = 40.0,
        width    = 512,
        height   = 512
    )

    return scene, camera



if __name__ == "__main__":
    np.random.seed(42)

    print("=== Трассировка путей ===")
    print("Построение сцены (Корнельская коробка)...")
    scene, camera = build_cornell_box()
    print(f"Треугольников: {len(scene.triangles)}")
    print(f"Источников:    {len(scene._lights)}")

    # Параметры рендера
    SPP       = 64   # лучей на пиксель
    MAX_DEPTH = 8      # максимальная глубина пути
    GAMMA     = 2.2
    EXPOSURE  = 1.0
    OUTPUT    = "cornell_box.ppm"

    print(f"Разрешение: {camera.width}x{camera.height}, SPP={SPP}")
    hdr = render(scene, camera,
                 spp=SPP,
                 max_depth=MAX_DEPTH,
                 gamma=GAMMA,
                 exposure=EXPOSURE,
                 output_path=OUTPUT)