import numpy as np
import os
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# Вспомогательные математические функции
# ──────────────────────────────────────────────

EPS = 1e-6
INF = 1e30


def luminance(c: np.ndarray) -> float:
    """Воспринимаемая яркость цвета (Rec. 709)."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Зеркальное отражение вектора d от нормали n."""
    return d - 2.0 * np.dot(d, n) * n

def cosine_sample_hemisphere(normal: np.ndarray) -> np.ndarray:
    """Выборка по косинусному закону (Ламберт) в полусфере вокруг normal."""
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


# ──────────────────────────────────────────────
# Структуры данных сцены
# ──────────────────────────────────────────────

@dataclass
class Material:
    """Материал поверхности."""
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
    """Один треугольник сетки."""
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material: Material

    def __post_init__(self):
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        n  = np.cross(e1, e2)
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
        """Алгоритм Мёллера–Трумбора. Возвращает t или None."""
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        h  = np.cross(ray_dir, e2)
        a  = np.dot(e1, h)
        if abs(a) < EPS:
            return None
        f  = 1.0 / a
        s  = ray_orig - self.v0
        u  = f * np.dot(s, h)
        if not (0.0 <= u <= 1.0):
            return None
        q  = np.cross(s, e1)
        v  = f * np.dot(ray_dir, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t  = f * np.dot(e2, q)
        return t if t > EPS else None


# ──────────────────────────────────────────────
# Сцена
# ──────────────────────────────────────────────

class Scene:
    def __init__(self):
        self.triangles: List[Triangle] = []
        self._lights:   List[Triangle] = []        # кэш источников
        self._light_areas: np.ndarray  = np.array([])
        self._light_powers: np.ndarray = np.array([])

    def add_triangle(self, tri: Triangle):
        self.triangles.append(tri)
        if tri.material.is_emitter:
            self._lights.append(tri)
        self._rebuild_light_cache()

    def add_triangles(self, tris: List[Triangle]):
        for t in tris:
            self.triangles.append(t)
            if t.material.is_emitter:
                self._lights.append(t)
        self._rebuild_light_cache()

    def _rebuild_light_cache(self):
        if not self._lights:
            return
        # Мощность источника ∝ площадь × сумма каналов emission
        powers = np.array([np.sum(t.material.emission) * t.area
                           for t in self._lights])
        self._light_powers = powers / powers.sum()

    # ── Пересечение ────────────────────────────
    def intersect(self, orig: np.ndarray, direction: np.ndarray
                  ) -> Tuple[Optional[Triangle], float]:
        """Ближайшее пересечение луча со сценой."""
        closest_t   = INF
        closest_tri = None
        for tri in self.triangles:
            t = tri.intersect(orig, direction)
            if t is not None and t < closest_t:
                closest_t   = t
                closest_tri = tri
        return closest_tri, closest_t

    def is_occluded(self, orig: np.ndarray, direction: np.ndarray,
                    max_t: float) -> bool:
        """Проверка видимости (тень)."""
        for tri in self.triangles:
            t = tri.intersect(orig, direction)
            if t is not None and t < max_t - EPS:
                return True
        return False

    # ── Выборка источника ──────────────────────
    def sample_light(self) -> Optional[Triangle]:
        if not self._lights:
            return None
        idx = np.random.choice(len(self._lights), p=self._light_powers)
        return self._lights[idx]

    def light_pdf(self, light: Triangle) -> float:
        """Полный pdf выборки точки на источнике: p_select / area."""
        for i, l in enumerate(self._lights):
            if l is light:
                return self._light_powers[i] / light.area
        return 0.0


# ──────────────────────────────────────────────
# Камера
# ──────────────────────────────────────────────

class Camera:
    def __init__(self, position: np.ndarray, look_at: np.ndarray,
                 up: np.ndarray, fov_deg: float,
                 width: int, height: int):
        self.position = position
        self.width    = width
        self.height   = height

        fwd = normalize(look_at - position)
        rgt = normalize(np.cross(fwd, up))
        u   = np.cross(rgt, fwd)

        half_h = np.tan(np.radians(fov_deg / 2.0))
        half_w = half_h * width / height

        self.lower_left = fwd - half_w * rgt - half_h * u
        self.horiz      = 2.0 * half_w * rgt
        self.vert       = 2.0 * half_h * u
        self.right      = rgt
        self.up         = u

    def get_ray(self, px: int, py: int
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (origin, direction) с антиалиасингом."""
        # Случайное смещение внутри пикселя
        sx = (px + np.random.random()) / self.width
        sy = (py + np.random.random()) / self.height
        direction = normalize(self.lower_left + sx * self.horiz + sy * self.vert)
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
        last_specular = True  # первый луч из камеры — считаем «зеркальным»

        for depth in range(self.max_depth):

            tri, t = self.scene.intersect(orig, direction)

            if tri is None:
                # Промахнулись — фон (чёрный)
                break

            hit_point = orig + t * direction
            mat       = tri.material
            normal    = tri.normal

            # Нормаль смотрит навстречу лучу
            if np.dot(normal, direction) > 0:
                normal = -normal

            # ── Эмиссия ────────────────────────
            if mat.is_emitter:
                # Учитываем эмиссию только если пришли по зеркальному пути,
                # т.к. для диффузных путей эмиссия уже учтена через NEE
                if last_specular:
                    color += throughput * mat.emission
                break

            # ── Прямое освещение (NEE) ─────────
            color += throughput * self._direct_light(hit_point, normal, mat)

            # ── Выбор события: диффузия или зеркало ──
            kd_lum = luminance(mat.diffuse)
            ks_lum = luminance(mat.specular)
            total  = kd_lum + ks_lum

            if total < EPS:
                break

            p_diff = kd_lum / total

            if np.random.random() < p_diff:
                # Диффузное рассеяние (Ламберт)
                new_dir = cosine_sample_hemisphere(normal)
                throughput = throughput * mat.diffuse / p_diff
                last_specular = False
            else:
                # Зеркальное отражение
                new_dir = reflect(direction, normal)
                throughput = throughput * mat.specular / (1.0 - p_diff)
                last_specular = True

            # ── Русская рулетка ────────────────
            if depth >= self.rr_start_depth:
                rr_prob = min(0.95, luminance(throughput))
                if rr_prob < EPS or np.random.random() > rr_prob:
                    break
                throughput /= rr_prob

            orig      = hit_point + normal * EPS
            direction = new_dir

        return color

    # ── Прямое освещение ──────────────────────
    def _direct_light(self, point: np.ndarray, normal: np.ndarray,
                      mat: Material) -> np.ndarray:
        """Выборка прямого освещения через один случайный источник."""
        light = self.scene.sample_light()
        if light is None:
            return np.zeros(3)

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
        contrib = light.material.emission * (mat.diffuse / np.pi) * geom / pdf_light
        return contrib


# ──────────────────────────────────────────────
# Многопроцессорный рендер
# ──────────────────────────────────────────────

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

    # ── Тональная компрессия ──────────────────
    img = hdr_buffer * exposure
    # Нормировка по средней яркости → 0.5
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

    # ── Запись PPM ───────────────────────────
    _save_ppm(img_uint8, output_path)
    print(f"Изображение сохранено: {output_path}")

    return hdr_buffer


def _save_ppm(img: np.ndarray, path: str):
    H, W, _ = img.shape
    with open(path, "wb") as f:
        header = f"P6\n{W} {H}\n255\n"
        f.write(header.encode())
        f.write(img.tobytes())


# ──────────────────────────────────────────────
# Загрузка OBJ (опционально)
# ──────────────────────────────────────────────

def load_obj(path: str, material: Material,
             scale: float = 1.0,
             offset: np.ndarray = None) -> List[Triangle]:
    """Минималистичный загрузчик OBJ (только v и f)."""
    if offset is None:
        offset = np.zeros(3)

    vertices: List[np.ndarray] = []
    triangles: List[Triangle]  = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                v = np.array([float(parts[1]),
                               float(parts[2]),
                               float(parts[3])]) * scale + offset
                vertices.append(v)
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Поддержка форматов: f v, f v/vt, f v/vt/vn
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                # Триангуляция (fan)
                for i in range(1, len(idxs) - 1):
                    tri = Triangle(vertices[idxs[0]].copy(),
                                   vertices[idxs[i]].copy(),
                                   vertices[idxs[i+1]].copy(),
                                   material)
                    triangles.append(tri)
    return triangles


# ──────────────────────────────────────────────
# Построение тестовой сцены — Корнельская коробка
# ──────────────────────────────────────────────

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
    light_mat = Material(emission=np.array([15.0, 15.0, 12.0]))

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

    # Маленький зеркальный «куб» (6 граней × 2 треугольника)
    def box(x0,y0,z0,x1,y1,z1, mat):
        tris = []
        tris += quad([x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1], mat)  # дно
        tris += quad([x0,y1,z0],[x0,y1,z1],[x1,y1,z1],[x1,y1,z0], mat)  # крыша
        tris += quad([x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1], mat)  # лево
        tris += quad([x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0], mat)  # право
        tris += quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0], mat)  # перед
        tris += quad([x0,y0,z1],[x0,y1,z1],[x1,y1,z1],[x1,y0,z1], mat)  # зад
        return tris

    # Высокий зеркальный блок
    scene.add_triangles(box(0.55, 0.0, 0.42,
                             0.82, 0.6, 0.70, mirror))
    # Низкий блок со смешанным материалом
    scene.add_triangles(box(0.18, 0.0, 0.18,
                             0.45, 0.3, 0.45, mixed))

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


# ──────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=== Трассировка путей ===")
    print("Построение сцены (Корнельская коробка)...")
    scene, camera = build_cornell_box()
    print(f"Треугольников: {len(scene.triangles)}")
    print(f"Источников:    {len(scene._lights)}")

    # ── Параметры рендера ──
    SPP       = 32   # лучей на пиксель
    MAX_DEPTH = 8      # максимальная глубина пути
    GAMMA     = 2.2
    EXPOSURE  = 1.0
    OUTPUT    = "cornell_box.ppm"

    print(f"Разрешение: {camera.width}×{camera.height}, SPP={SPP}")
    hdr = render(scene, camera,
                 spp=SPP,
                 max_depth=MAX_DEPTH,
                 gamma=GAMMA,
                 exposure=EXPOSURE,
                 output_path=OUTPUT)

    # Опционально: сохранить HDR в бинарный формат
    hdr.astype(np.float32).tofile(OUTPUT.replace(".ppm", ".hdr_raw"))
    print("HDR-данные сохранены (float32 raw).")