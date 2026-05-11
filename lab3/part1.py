import numpy as np

def sample_triangle(V1, V2, V3, N=100000):
    r1 = np.random.rand(N)
    r2 = np.random.rand(N)

    sqrt_r1 = np.sqrt(r1)

    P = (1 - sqrt_r1)[:, None] * V1 + \
        (sqrt_r1 * (1 - r2))[:, None] * V2 + \
        (sqrt_r1 * r2)[:, None] * V3
    return P

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

def sample_sphere(N=100000):
    u1 = np.random.rand(N)
    u2 = np.random.rand(N)

    phi = 2*np.pi*u1
    z = 1 - 2*u2
    r = np.sqrt(1 - z**2)

    x = r*np.cos(phi)
    y = r*np.sin(phi)

    return np.stack([x,y,z], axis=1)

def sample_cosine_hemisphere(N=100000):
    u1 = np.random.rand(N)
    u2 = np.random.rand(N)

    r = np.sqrt(u1)
    phi = 2*np.pi*u2

    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z = np.sqrt(1 - r**2)

    return np.stack([x,y,z], axis=1)