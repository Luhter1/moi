"""
Lab 5: Joint Bilateral Filter для подавления шума в Path Tracing.

Архитектура:
1. Модифицированный рендерер — трассировка путей с сохранением G-буферов
   (direct_light, indirect_light, depth_map, normal_map, object_index).
2. Joint Bilateral Filter — фильтрация только indirect_light с использованием
   G-буферов в качестве весов, предотвращающих размытие границ.
3. Тональная компрессия, гамма-коррекция, сохранение результата.

Физическая корректность:
- Вся обработка ведётся в линейном RGB пространстве.
- Гамма-коррекция применяется только после фильтрации.
- Энергосбережение обеспечивается нормировкой весов фильтра.
"""

import numpy as np
import os
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

EPS = 1e-6
INF = 1e30


# ---------------------------------------------------------------------------
# Вспомогательные математические функции
# ---------------------------------------------------------------------------

def luminance(c: np.ndarray) -> float:
    """Воспринимаемая яркость цвета (линейный RGB -> яркость)."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else v


def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Зеркальное отражение вектора d от нормали n."""
    return d - 2.0 * np.dot(d, n) * n


def cosine_sample_hemisphere(normal: np.ndarray) -> np.ndarray:
    """
    Генерирует случайное направление отражения для диффузных поверхностей
    согласно закону Ламберта (косинусное распределение).
    """
    u1, u2 = np.random.random(), np.random.random()
    r = np.sqrt(u1)
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(max(0.0, 1.0 - u1))

    up = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = normalize(np.cross(up, normal))
    b = np.cross(normal, t)
    return x * t + y * b + z * normal


# ---------------------------------------------------------------------------
# Структуры данных сцены
# ---------------------------------------------------------------------------

@dataclass
class Material:
    diffuse: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))
    specular: np.ndarray = field(default_factory=lambda: np.zeros(3))
    emission: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        total = self.diffuse + self.specular
        mask = total > 1.0
        if np.any(mask):
            scale = np.where(mask, 1.0 / total, 1.0)
            self.diffuse = self.diffuse * scale
            self.specular = self.specular * scale

    @property
    def is_emitter(self) -> bool:
        return np.any(self.emission > 0.0)


@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material: Material
    tri_index: int = 0  # Уникальный индекс треугольника (для object_index)

    def __post_init__(self):
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        n = np.cross(self.e1, self.e2)
        self.area = 0.5 * np.linalg.norm(n)
        self.normal = normalize(n)

    def sample_point(self) -> np.ndarray:
        u1, u2 = np.random.random(), np.random.random()
        if u1 + u2 > 1.0:
            u1, u2 = 1.0 - u1, 1.0 - u2
        return self.v0 + u1 * (self.v1 - self.v0) + u2 * (self.v2 - self.v0)

    def intersect(self, ray_orig: np.ndarray, ray_dir: np.ndarray
                  ) -> Optional[float]:
        h = np.cross(ray_dir, self.e2)
        a = np.dot(self.e1, h)
        if abs(a) < EPS:
            return None
        f = 1.0 / a
        s = ray_orig - self.v0
        u = f * np.dot(s, h)
        if not (0.0 <= u <= 1.0):
            return None
        q = np.cross(s, self.e1)
        v = f * np.dot(ray_dir, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(self.e2, q)
        return t if t > EPS else None


# ---------------------------------------------------------------------------
# Сцена
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self):
        self.triangles: List[Triangle] = []
        self._lights: List[Triangle] = []
        self._light_powers: np.ndarray = np.array([])
        self._light_pdf_map: dict = {}
        self._accel_built = False
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
        self._rebuild_light_cache()
        N = len(self.triangles)
        self._n_tri = N
        if N == 0:
            return
        self._v0f = np.array([t.v0 for t in self.triangles]).ravel().astype(np.float64)
        self._e1f = np.array([t.e1 for t in self.triangles]).ravel().astype(np.float64)
        self._e2f = np.array([t.e2 for t in self.triangles]).ravel().astype(np.float64)
        self._accel_built = True

    def intersect(self, orig: np.ndarray, direction: np.ndarray
                  ) -> Tuple[Optional[Triangle], float]:
        d = direction
        v0f = self._v0f; e1f = self._e1f; e2f = self._e2f

        hx = d[1] * e2f[2::3] - d[2] * e2f[1::3]
        hy = d[2] * e2f[0::3] - d[0] * e2f[2::3]
        hz = d[0] * e2f[1::3] - d[1] * e2f[0::3]

        a = e1f[0::3] * hx + e1f[1::3] * hy + e1f[2::3] * hz
        valid = np.abs(a) > EPS
        if not np.any(valid):
            return None, INF

        a_safe = np.where(valid, a, 1.0)
        f = 1.0 / a_safe
        f[~valid] = 0.0

        sx = orig[0] - v0f[0::3]
        sy = orig[1] - v0f[1::3]
        sz = orig[2] - v0f[2::3]

        u = f * (sx * hx + sy * hy + sz * hz)
        valid &= (u >= 0.0) & (u <= 1.0)

        qx = sy * e1f[2::3] - sz * e1f[1::3]
        qy = sz * e1f[0::3] - sx * e1f[2::3]
        qz = sx * e1f[1::3] - sy * e1f[0::3]

        v = f * (d[0] * qx + d[1] * qy + d[2] * qz)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        t = f * (e2f[0::3] * qx + e1f[1::3] * qy + e2f[2::3] * qz)
        valid &= (t > EPS)

        if not np.any(valid):
            return None, INF

        t[~valid] = INF
        idx = int(np.argmin(t))
        return self.triangles[idx], t[idx]

    def is_occluded(self, orig: np.ndarray, direction: np.ndarray,
                    max_t: float) -> bool:
        d = direction
        v0f = self._v0f; e1f = self._e1f; e2f = self._e2f

        hx = d[1] * e2f[2::3] - d[2] * e2f[1::3]
        hy = d[2] * e2f[0::3] - d[0] * e2f[2::3]
        hz = d[0] * e2f[1::3] - d[1] * e2f[0::3]

        a = e1f[0::3] * hx + e1f[1::3] * hy + e1f[2::3] * hz
        valid = np.abs(a) > EPS
        if not np.any(valid):
            return False

        a_safe = np.where(valid, a, 1.0)
        f = 1.0 / a_safe
        f[~valid] = 0.0

        sx = orig[0] - v0f[0::3]
        sy = orig[1] - v0f[1::3]
        sz = orig[2] - v0f[2::3]

        u = f * (sx * hx + sy * hy + sz * hz)
        valid &= (u >= 0.0) & (u <= 1.0)

        qx = sy * e1f[2::3] - sz * e1f[1::3]
        qy = sz * e1f[0::3] - sx * e1f[2::3]
        qz = sx * e1f[1::3] - sy * e1f[0::3]

        v = f * (d[0] * qx + d[1] * qy + d[2] * qz)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        t = f * (e2f[0::3] * qx + e1f[1::3] * qy + e2f[2::3] * qz)
        valid &= (t > EPS) & (t < max_t - EPS)

        return bool(np.any(valid))

    def sample_light(self) -> Optional[Triangle]:
        if not self._lights:
            return None
        idx = np.random.choice(len(self._lights), p=self._light_powers)
        return self._lights[idx]

    def light_pdf(self, light: Triangle) -> float:
        idx = self._light_pdf_map.get(id(light))
        if idx is None:
            return 0.0
        return self._light_powers[idx] / light.area


# ---------------------------------------------------------------------------
# Камера
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, position: np.ndarray, look_at: np.ndarray,
                 up: np.ndarray, fov_deg: float,
                 width: int, height: int):
        self.position = position
        self.width = width
        self.height = height

        fwd = normalize(look_at - position)
        rgt = normalize(np.cross(fwd, up))
        u = np.cross(rgt, fwd)

        half_h = np.tan(np.radians(fov_deg / 2.0))
        half_w = half_h * width / height

        self.lower_left = fwd - half_w * rgt - half_h * u
        self.horiz = 2.0 * half_w * rgt
        self.vert = 2.0 * half_h * u
        self.right = rgt
        self.up = u

    def get_ray(self, px: int, py: int
                ) -> Tuple[np.ndarray, np.ndarray]:
        sx = (px + np.random.random()) / self.width
        sy = (py + np.random.random()) / self.height
        direction = normalize(self.lower_left + sx * self.horiz + sy * self.vert)
        return self.position.copy(), direction


# ---------------------------------------------------------------------------
# Трассировщик путей с G-буферами
# ---------------------------------------------------------------------------

class PathTracerGBuf:
    """
    Модифицированный трассировщик путей, который разделяет вклад на:
    - direct_light: вклад от NEE (прямое освещение через явную выборку источника)
    - indirect_light: всё остальное (эмиссия от случайных попаданий, specular, и т.д.)
    """

    def __init__(self, scene: Scene, camera: Camera,
                 max_depth: int = 8,
                 rr_start_depth: int = 3):
        self.scene = scene
        self.camera = camera
        self.max_depth = max_depth
        self.rr_start_depth = rr_start_depth

    def trace(self, orig: np.ndarray, direction: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Трассировка одного пути. Возвращает (direct, indirect) вклады.

        Разделение:
        - direct: результат _direct_light() (NEE вклад)
        - indirect: эмиссия при попадании на источник через случайныйbounce,
                    specular отражения, и прочие некомпоненты прямого света
        """
        direct_color = np.zeros(3)
        indirect_color = np.zeros(3)
        throughput = np.ones(3)
        nee_value = np.zeros(3)
        has_nee = False

        for depth in range(self.max_depth):
            tri, t = self.scene.intersect(orig, direction)

            if tri is None:
                break

            hit_point = orig + t * direction
            mat = tri.material
            normal = tri.normal

            if np.dot(normal, direction) > 0:
                normal = -normal

            # Эмиссия (источник света, обнаруженный через случайный bounce)
            if mat.is_emitter:
                if has_nee:
                    # Компенсируем двойной счёт: если мы уже учли этот вклад
                    # через NEE, вычитаем его из indirect
                    indirect_color -= nee_value
                indirect_color += throughput * mat.emission
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
                # NEE: прямое освещение -> direct
                nee_value = throughput * self._direct_light(hit_point, normal, mat)
                direct_color += nee_value
                has_nee = True

                # Продолжаем путь (косвенное освещение)
                new_dir = cosine_sample_hemisphere(normal)
                throughput = throughput * mat.diffuse
            else:
                # Specular: вклад в indirect
                new_dir = reflect(direction, normal)
                throughput = throughput * mat.specular
                has_nee = False
                nee_value = np.zeros(3)

            orig = hit_point + normal * EPS
            direction = new_dir

        return direct_color, indirect_color

    def _direct_light(self, point: np.ndarray, normal: np.ndarray,
                      mat: Material) -> np.ndarray:
        """Прямое освещение через явную выборку источника (NEE)."""
        light = self.scene.sample_light()
        if light is None:
            return np.zeros(3)

        light_point = light.sample_point()
        to_light = light_point - point
        dist = np.linalg.norm(to_light)
        if dist < EPS:
            return np.zeros(3)
        to_light_n = to_light / dist

        cos_surf = np.dot(normal, to_light_n)
        cos_light = np.dot(-light.normal, to_light_n)

        if cos_surf <= 0 or cos_light <= 0:
            return np.zeros(3)

        if self.scene.is_occluded(point + normal * EPS, to_light_n, dist):
            return np.zeros(3)

        pdf_light = self.scene.light_pdf(light)
        if pdf_light < EPS:
            return np.zeros(3)

        geom = cos_surf * cos_light / (dist * dist)
        contrib = light.material.emission * (mat.diffuse / np.pi) * geom / pdf_light
        return contrib


# ---------------------------------------------------------------------------
# Рендеринг с G-буферами
# ---------------------------------------------------------------------------

# Глобальные переменные воркеров
_w_tracer: Optional[PathTracerGBuf] = None
_w_camera: Optional[Camera] = None


def _worker_init(tracer: PathTracerGBuf, camera: Camera):
    global _w_tracer, _w_camera
    _w_tracer = tracer
    _w_camera = camera
    np.random.seed(os.getpid())


def _render_row_gbuf(args: Tuple[int, int]) -> Tuple[int, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]:
    """
    Рендерит одну строку, возвращая:
    py, direct_row, indirect_row, depth_row, normal_row, obj_idx_row
    """
    py, spp = args
    W = _w_camera.width

    direct_row = np.zeros((W, 3))
    indirect_row = np.zeros((W, 3))
    depth_row = np.full(W, INF)
    normal_row = np.zeros((W, 3))
    obj_idx_row = np.full(W, -1, dtype=np.int32)

    for px in range(W):
        d_acc = np.zeros(3)
        i_acc = np.zeros(3)

        # Первый сэмпл — заполнить G-буферы
        first_orig, first_dir = _w_camera.get_ray(px, py)
        tri, t = _w_tracer.scene.intersect(first_orig, first_dir)
        if tri is not None:
            depth_row[px] = t
            n = tri.normal
            if np.dot(n, first_dir) > 0:
                n = -n
            normal_row[px] = n
            obj_idx_row[px] = tri.tri_index

        # Накопление всех сэмплов
        d_acc, i_acc = _w_tracer.trace(first_orig, first_dir)
        total_d = d_acc.copy()
        total_i = i_acc.copy()

        for _ in range(1, spp):
            orig, direction = _w_camera.get_ray(px, py)
            d, i = _w_tracer.trace(orig, direction)
            total_d += d
            total_i += i

        direct_row[px] = total_d / spp
        indirect_row[px] = total_i / spp

    return py, direct_row, indirect_row, depth_row, normal_row, obj_idx_row


def render_with_gbuf(scene: Scene, camera: Camera,
                     spp: int = 64,
                     max_depth: int = 8) -> dict:
    """
    Рендерит сцену и возвращает словарь с G-буферами.

    Returns:
        dict с ключами:
            'direct_light'   — (H, W, 3) прямое освещение
            'indirect_light' — (H, W, 3) косвенное освещение
            'depth_map'      — (H, W) глубина
            'normal_map'     — (H, W, 3) нормали
            'object_index'   — (H, W) индекс объекта
    """
    W, H = camera.width, camera.height
    scene.build_accel()
    tracer = PathTracerGBuf(scene, camera, max_depth=max_depth)

    direct_buf = np.zeros((H, W, 3), dtype=np.float64)
    indirect_buf = np.zeros((H, W, 3), dtype=np.float64)
    depth_buf = np.full((H, W), INF, dtype=np.float64)
    normal_buf = np.zeros((H, W, 3), dtype=np.float64)
    objidx_buf = np.full((H, W), -1, dtype=np.int32)

    start_time = time.time()
    ncpus = mp.cpu_count()

    if ncpus > 1:
        print(f"  Используем {ncpus} процессов")
        with mp.Pool(ncpus, initializer=_worker_init,
                     initargs=(tracer, camera)) as pool:
            for i, result in enumerate(
                    pool.imap_unordered(_render_row_gbuf,
                                        [(py, spp) for py in range(H)])):
                py, d_row, i_row, z_row, n_row, o_row = result
                direct_buf[py] = d_row
                indirect_buf[py] = i_row
                depth_buf[py] = z_row
                normal_buf[py] = n_row
                objidx_buf[py] = o_row

                pct = (i + 1) * 100 // H
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (H - i - 1)
                print(f"\r  Строка {i+1:4d}/{H} ({pct:3d}%)  "
                      f"ETA {eta:6.1f} с   ", end="", flush=True)
    else:
        for py in range(H):
            result = _render_row_gbuf((py, spp))
            _, d_row, i_row, z_row, n_row, o_row = result
            direct_buf[py] = d_row
            indirect_buf[py] = i_row
            depth_buf[py] = z_row
            normal_buf[py] = n_row
            objidx_buf[py] = o_row

            elapsed = time.time() - start_time
            done = (py + 1) * W
            eta = elapsed / done * (W * H - done) if done else 0
            print(f"\r  Строка {py+1:4d}/{H}  ETA {eta:6.1f} с   ",
                  end="", flush=True)

    print(f"\nРендер завершён за {time.time() - start_time:.1f} с")

    return {
        'direct_light': direct_buf,
        'indirect_light': indirect_buf,
        'depth_map': depth_buf,
        'normal_map': normal_buf,
        'object_index': objidx_buf,
    }


# ---------------------------------------------------------------------------
# Joint Bilateral Filter
# ---------------------------------------------------------------------------

def joint_bilateral_filter(
    noisy_image: np.ndarray,
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    object_index: np.ndarray,
    sigma_s: float = 4.0,
    sigma_z: float = 0.05,
    sigma_n: float = 0.15,
    radius: int = 8,
) -> np.ndarray:
    """
    Joint/Cross Bilateral Filter с использованием G-буферов.

    Фильтрует только зашумлённое изображение (обычно indirect_light),
    используя пространственные и диапазонные веса из G-буферов.

    Математическая модель:
    ---------------------------------------------------------------------------
    Для каждого пикселя p вычисляем:

        g(p) = (1 / W_p) * Σ_{q ∈ S} f(q) * G_s(||p - q||) * G_r(p, q)

    где:
        f(q)        — цвет пикселя q из зашумлённого изображения
        G_s(||p-q||) — пространственный вес (гауссиан от расстояния)
        G_r(p,q)    — диапазонный вес, зависящий от G-буферов

    Пространственный вес:
        G_s(||p - q||) = exp(-||p - q||² / (2 * σ_s²))

    Диапазонный вес (произведение трёх компонент):
        G_r(p, q) = G_r_obj(p, q) * G_r_depth(p, q) * G_r_normal(p, q)

    1. G_r_obj(p, q) — индекс объекта:
       = 0, если object_index[p] ≠ object_index[q]
         (никогда не смешиваем цвета разных объектов — жёсткая граница)
       = 1, иначе

    2. G_r_depth(p, q) — разница глубин:
       = exp(-(depth_map[p] - depth_map[q])² / (2 * σ_z²))
       Вес падает при большом перепаде глубины, сохраняя границы объектов,
       расположенных на разных расстояниях от камеры.

    3. G_r_normal(p, q) — разница нормалей:
       = exp(-(1 - dot(normal_map[p], normal_map[q])) / (2 * σ_n²))
       Используем (1 - dot(n_p, n_q)) как меру угла между нормалями.
       Для совпадающих нормалей dot = 1, мера = 0 (максимальный вес).
       Для ортогональных нормалей dot = 0, мера = 1 (вес падает).
       Это сохраняет геометрические границы (рёбра, углы).

    Нормировка (закон сохранения энергии):
    ---------------------------------------------------------------------------
        W_p = Σ_{q ∈ S} G_s(||p - q||) * G_r(p, q)

    После деления на W_p каждый пиксель g(p) является взвешенным средним
    соседей с весами, сумма которых равна 1. Это гарантирует:
        - Локальное сохранение энергии: яркость не добавляется и не удаляется,
          а только перераспределяется между соседними пикселями.
        - Глобальное сохранение: Σ_p g(p) ≈ Σ_p f(p) с высокой точностью
          (небольшие отклонения только на границах изображения из-за padding).

    Чек-лист энергосбережения:
    ---------------------------------------------------------------------------
    [✓] Веса нормируются на сумму W_p для каждого пикселя
    [✓] G_s — гауссиан, всегда ≥ 0
    [✓] G_r_obj ∈ {0, 1}, никогда не отрицателен
    [✓] G_r_depth = exp(...) > 0, всегда положителен
    [✓] G_r_normal = exp(...) > 0, всегда положителен
    [✓] Все веса ≥ 0 ⇒ W_p > 0 (при ненулевом ядре) ⇒ нормировка корректна
    [✓] g(p) = взвешенное среднее с Σ weights = 1 ⇒ энергия сохраняется

    Параметры:
    ---------------------------------------------------------------------------
        sigma_s : float — пространственная сигма гауссиана.
            Определяет размер области фильтрации.
            Больше σ_s → сильнее размытие шума, но могут замыться детали.
            Рекомендуемый диапазон: 2–8 для разрешения 512×512.

        sigma_z : float — сигма для разницы глубин.
            Определяет, насколько чувствителен фильтр к перепадам глубины.
            Меньше σ_z → более резкие границы по глубине.
            Для сцены с глубиной ~0–2 ед.: рекомендуемый диапазон 0.02–0.1.

        sigma_n : float — сигма для разницы нормалей.
            Определяет, насколько чувствителен фильтр к изменению нормалей.
            Меньше σ_n → более резкие границы по геометрии.
            Рекомендуемый диапазон: 0.05–0.3.

        radius : int — радиус ядра фильтра.
            Ядро имеет размер (2*radius + 1) × (2*radius + 1).
            Рекомендуется radius ≈ 2 × sigma_s.
    """
    H, W, C = noisy_image.shape

    # ---------------------------------------------------------------------------
    # 1. Предвычисление пространственного гауссиана G_s
    # ---------------------------------------------------------------------------
    # Создаём ядро G_s размера (2r+1) × (2r+1)
    # G_s(dx, dy) = exp(-(dx² + dy²) / (2 * σ_s²))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    spatial_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_s ** 2))
    # Форма: (kernel_size, kernel_size), kernel_size = 2*radius + 1

    # ---------------------------------------------------------------------------
    # 2. Подготовка данных — заменяем невалидные пиксели (нет попадания луча)
    # ---------------------------------------------------------------------------
    # Нормализуем normal_map (на всякий случай)
    normal_norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_norms = np.where(normal_norms < EPS, 1.0, normal_norms)
    normals = normal_map / normal_norms

    # Для depth_map: помечаем пиксели без попадания как бесконечность
    depth = depth_map.copy()

    # Для object_index: помечаем -1 как уникальный индекс (не совпадёт ни с чем)
    obj_idx = object_index.copy()

    # ---------------------------------------------------------------------------
    # 3. Pad массивы для удобства индексации
    # ---------------------------------------------------------------------------
    pad = radius
    noisy_pad = np.pad(noisy_image, ((pad, pad), (pad, pad), (0, 0)),
                        mode='edge')
    depth_pad = np.pad(depth, ((pad, pad), (pad, pad)), mode='edge')
    normals_pad = np.pad(normals, ((pad, pad), (pad, pad), (0, 0)),
                          mode='edge')
    objidx_pad = np.pad(obj_idx, ((pad, pad), (pad, pad)), mode='edge')

    # ---------------------------------------------------------------------------
    # 4. Основной цикл фильтрации
    # ---------------------------------------------------------------------------
    # Итоговое изображение и карта весов
    result = np.zeros_like(noisy_image, dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    kernel_size = 2 * radius + 1

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            # Пространственный вес G_s для этого смещения
            gs = spatial_kernel[dy, dx]

            # Срезы для пикселей p и их соседей q = p + (dx - radius, dy - radius)
            # q-область в padded массиве
            q_y_start = dy
            q_y_end = dy + H
            q_x_start = dx
            q_x_end = dx + W

            # Соседние значения
            f_q = noisy_pad[q_y_start:q_y_end, q_x_start:q_x_end]  # (H, W, 3)
            depth_q = depth_pad[q_y_start:q_y_end, q_x_start:q_x_end]  # (H, W)
            normals_q = normals_pad[q_y_start:q_y_end, q_x_start:q_x_end]  # (H, W, 3)
            objidx_q = objidx_pad[q_y_start:q_y_end, q_x_start:q_x_end]  # (H, W)

            # ---------------------------------------------------------------------------
            # Вычисление диапазонных весов G_r
            # ---------------------------------------------------------------------------

            # G_r_obj: жёсткая маска по индексу объекта
            # = 0 если разные объекты, = 1 если один объект
            gr_obj = (obj_idx == objidx_q).astype(np.float64)

            # G_r_depth: гауссиан от разницы глубин
            # exp(-(depth_p - depth_q)² / (2 * σ_z²))
            depth_diff = depth - depth_q
            gr_depth = np.exp(-(depth_diff ** 2) / (2.0 * sigma_z ** 2))

            # G_r_normal: гауссиан от угла между нормалями
            # exp(-(1 - dot(n_p, n_q)) / (2 * σ_n²))
            # dot product по каналам: sum(n_p * n_q, axis=2)
            dot_pn = np.sum(normals * normals_q, axis=2)  # (H, W)
            # Ограничиваем dot product в [−1, 1] для численной стабильности
            dot_pn = np.clip(dot_pn, -1.0, 1.0)
            gr_normal = np.exp(-(1.0 - dot_pn) / (2.0 * sigma_n ** 2))

            # Итоговый диапазонный вес
            gr = gr_obj * gr_depth * gr_normal  # (H, W)

            # Полный вес для этого смещения: G_s * G_r
            w = gs * gr  # (H, W)

            # Накопление взвешенной суммы
            # f(q) * w, разворачиваем w до (H, W, 1) для broadcasts
            result += f_q * w[:, :, np.newaxis]
            weight_sum += w

    # ---------------------------------------------------------------------------
    # 5. Нормировка (закон сохранения энергии)
    # ---------------------------------------------------------------------------
    # W_p = Σ G_s * G_r для каждого пикселя
    # g(p) = result / W_p
    # Гарантирует, что g(p) — взвешенное среднее с Σ весов = 1
    weight_sum = np.where(weight_sum < EPS, 1.0, weight_sum)
    result = result / weight_sum[:, :, np.newaxis]

    return result


# ---------------------------------------------------------------------------
# Тональная компрессия и сохранение
# ---------------------------------------------------------------------------

def tone_map_and_save(hdr: np.ndarray, output_path: str,
                      gamma: float = 2.2, exposure: float = 1.0):
    """
    Тональная компрессия, гамма-коррекция и сохранение.

    Работает в линейном RGB, гамма применяется в самом конце.
    """
    img = hdr * exposure
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    nonzero = lum[lum > EPS]
    if len(nonzero) > 0:
        mean_lum = np.mean(nonzero)
        img *= 0.5 / mean_lum
    np.clip(img, 0.0, 1.0, out=img)

    img_gamma = np.power(img, 1.0 / gamma)
    img_uint8 = (img_gamma * 255.0).astype(np.uint8)

    _save_ppm(img_uint8, output_path)
    print(f"Изображение сохранено: {output_path}")

    # Попытка сохранить PNG через PIL (если доступна)
    try:
        from PIL import Image
        png_path = output_path.rsplit('.', 1)[0] + '.png'
        Image.fromarray(img_uint8).save(png_path)
        print(f"PNG сохранён: {png_path}")
    except ImportError:
        pass


def _save_ppm(img: np.ndarray, path: str):
    H, W, _ = img.shape
    with open(path, "wb") as f:
        header = f"P6\n{W} {H}\n255\n"
        f.write(header.encode())
        f.write(img.tobytes())


# ---------------------------------------------------------------------------
# Построение тестовой сцены — Корнельская коробка
# ---------------------------------------------------------------------------

def build_cornell_box() -> Tuple[Scene, Camera]:
    scene = Scene()

    white = Material(diffuse=np.array([0.73, 0.73, 0.73]))
    red = Material(diffuse=np.array([0.65, 0.05, 0.05]))
    green = Material(diffuse=np.array([0.12, 0.45, 0.15]))
    mirror = Material(diffuse=np.zeros(3),
                      specular=np.array([0.95, 0.95, 0.95]))
    mixed = Material(diffuse=np.array([0.5, 0.4, 0.1]),
                     specular=np.array([0.4, 0.4, 0.4]))
    light_mat = Material(emission=np.array([8.0, 8.0, 6.5]))

    tri_counter = 0

    def quad(v0, v1, v2, v3, mat):
        nonlocal tri_counter
        t1 = Triangle(np.array(v0), np.array(v1), np.array(v2), mat,
                       tri_index=tri_counter)
        tri_counter += 1
        t2 = Triangle(np.array(v0), np.array(v2), np.array(v3), mat,
                       tri_index=tri_counter)
        tri_counter += 1
        return [t1, t2]

    # Пол
    scene.add_triangles(quad(
        [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], white))
    # Потолок
    scene.add_triangles(quad(
        [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], white))
    # Задняя стена
    scene.add_triangles(quad(
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], white))
    # Левая стена (красная)
    scene.add_triangles(quad(
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], red))
    # Правая стена (зелёная)
    scene.add_triangles(quad(
        [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], green))

    # Источник света на потолке
    scene.add_triangles(quad(
        [0.35, 0.999, 0.35], [0.65, 0.999, 0.35],
        [0.65, 0.999, 0.65], [0.35, 0.999, 0.65], light_mat))

    def rotated_box(x0, y0, z0, x1, y1, z1, mat, angle_deg=0):
        cx = (x0 + x1) / 2
        cz = (z0 + z1) / 2
        a = np.radians(angle_deg)
        ca, sa = np.cos(a), np.sin(a)

        def rot(v):
            dx, dz = v[0] - cx, v[2] - cz
            return [cx + dx * ca - dz * sa, v[1], cz + dx * sa + dz * ca]

        tris = []
        tris += quad(rot([x0, y0, z0]), rot([x1, y0, z0]),
                     rot([x1, y0, z1]), rot([x0, y0, z1]), mat)
        tris += quad(rot([x0, y1, z0]), rot([x0, y1, z1]),
                     rot([x1, y1, z1]), rot([x1, y1, z0]), mat)
        tris += quad(rot([x0, y0, z0]), rot([x0, y0, z1]),
                     rot([x0, y1, z1]), rot([x0, y1, z0]), mat)
        tris += quad(rot([x1, y0, z0]), rot([x1, y1, z0]),
                     rot([x1, y1, z1]), rot([x1, y0, z1]), mat)
        tris += quad(rot([x0, y0, z0]), rot([x0, y1, z0]),
                     rot([x1, y1, z0]), rot([x1, y0, z0]), mat)
        tris += quad(rot([x0, y0, z1]), rot([x1, y0, z1]),
                     rot([x1, y1, z1]), rot([x0, y1, z1]), mat)
        return tris

    scene.add_triangles(rotated_box(0.55, 0.0, 0.42,
                                    0.82, 0.6, 0.70, mirror, angle_deg=-30))
    scene.add_triangles(rotated_box(0.18, 0.0, 0.18,
                                    0.45, 0.3, 0.45, mixed, angle_deg=20))

    camera = Camera(
        position=np.array([0.5, 0.5, -1.4]),
        look_at=np.array([0.5, 0.5, 0.5]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=40.0,
        width=512,
        height=512
    )

    return scene, camera


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Lab 5: Joint Bilateral Filter для Path Tracing")
    print("=" * 60)

    # Параметры рендера
    SPP = 64
    MAX_DEPTH = 8
    GAMMA = 2.2

    # Параметры билатерального фильтра
    SIGMA_S = 4.0    # Пространственная сигма
    SIGMA_Z = 0.05   # Сигма глубины
    SIGMA_N = 0.15   # Сигма нормалей
    RADIUS = 8       # Радиус ядра (≈ 2 × sigma_s)

    print(f"\n--- Рендеринг с G-буферами (SPP={SPP}) ---")
    print("Построение сцены (Корнельская коробка)...")
    scene, camera = build_cornell_box()
    print(f"Треугольников: {len(scene.triangles)}")
    print(f"Источников:    {len(scene._lights)}")

    # Шаг 1: Рендеринг с G-буферами
    gbuf = render_with_gbuf(scene, camera, spp=SPP, max_depth=MAX_DEPTH)

    direct = gbuf['direct_light']
    indirect = gbuf['indirect_light']
    depth_map = gbuf['depth_map']
    normal_map = gbuf['normal_map']
    obj_idx = gbuf['object_index']

    # Исходное (незафильтированное) изображение
    total_unfiltered = direct + indirect

    print(f"\n--- Сохранение исходного изображения ---")
    tone_map_and_save(total_unfiltered, "cornell_unfiltered.ppm", gamma=GAMMA)

    # Шаг 2: Joint Bilateral Filter для indirect_light
    print(f"\n--- Joint Bilateral Filter ---")
    print(f"Параметры: σ_s={SIGMA_S}, σ_z={SIGMA_Z}, σ_n={SIGMA_N}, "
          f"radius={RADIUS}")
    print(f"Размер ядра: {2 * RADIUS + 1}×{2 * RADIUS + 1}")

    t_start = time.time()
    filtered_indirect = joint_bilateral_filter(
        noisy_image=indirect,
        depth_map=depth_map,
        normal_map=normal_map,
        object_index=obj_idx,
        sigma_s=SIGMA_S,
        sigma_z=SIGMA_Z,
        sigma_n=SIGMA_N,
        radius=RADIUS,
    )
    t_elapsed = time.time() - t_start
    print(f"\nФильтрация завершена за {t_elapsed:.1f} с")

    # Шаг 3: Объединяем direct + filtered indirect
    total_filtered = direct + filtered_indirect

    print(f"\n--- Сохранение отфильтрованного изображения ---")
    tone_map_and_save(total_filtered, "cornell_filtered.ppm", gamma=GAMMA)

    # Шаг 4: Проверка энергосбережения
    print(f"\n--- Проверка энергосбережения ---")
    energy_before = np.sum(total_unfiltered)
    energy_after = np.sum(total_filtered)
    energy_ratio = energy_after / energy_before if energy_before > 0 else 0
    print(f"Энергия до фильтрации:  {energy_before:.4f}")
    print(f"Энергия после фильтрации: {energy_after:.4f}")
    print(f"Отношение (после/до):  {energy_ratio:.6f}")
    if abs(energy_ratio - 1.0) < 0.01:
        print("[✓] Энергосбережение выполняется (отклонение < 1%)")
    else:
        print("[!] Отклонение энергосбережения > 1% — проверьте параметры")

    print("\nГотово!")
