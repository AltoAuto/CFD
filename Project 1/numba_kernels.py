import numpy as np
from numba import njit


@njit(cache=True)
def advect_scalar_numba(phi, flux_i, flux_j, volume):
    """Compute first-order upwind advection (Numba)."""
    count_i, count_j = volume.shape
    out = np.zeros((count_i, count_j), dtype=np.float64)
    for i in range(count_i):
        for j in range(count_j):
            fw = flux_i[i, j]
            fe = flux_i[i + 1, j]
            fs = flux_j[i, j]
            fn = flux_j[i, j + 1]

            if fw >= 0.0:
                phi_w = phi[i, j + 1]
            else:
                phi_w = phi[i + 1, j + 1]
            if fe >= 0.0:
                phi_e = phi[i + 1, j + 1]
            else:
                phi_e = phi[i + 2, j + 1]
            if fs >= 0.0:
                phi_s = phi[i + 1, j]
            else:
                phi_s = phi[i + 1, j + 1]
            if fn >= 0.0:
                phi_n = phi[i + 1, j + 1]
            else:
                phi_n = phi[i + 1, j + 2]

            div = fe * phi_e - fw * phi_w + fn * phi_n - fs * phi_s
            out[i, j] = -div / volume[i, j]
    return out


@njit(cache=True)
def grad_scalar_numba(
    phi,
    w_left_i,
    w_right_i,
    w_south_j,
    w_north_j,
    face_normal_i,
    face_normal_j,
    volume,
):
    """Compute cell-centered gradient of a scalar (Numba)."""
    count_i, count_j = volume.shape
    grad = np.zeros((count_i, count_j, 2), dtype=np.float64)
    for i in range(count_i):
        for j in range(count_j):
            wl = w_left_i[i, j]
            wr = w_right_i[i, j]
            phi_w = wl * phi[i, j + 1] + wr * phi[i + 1, j + 1]
            nwx = face_normal_i[i, j, 0]
            nwy = face_normal_i[i, j, 1]

            wl = w_left_i[i + 1, j]
            wr = w_right_i[i + 1, j]
            phi_e = wl * phi[i + 1, j + 1] + wr * phi[i + 2, j + 1]
            nex = face_normal_i[i + 1, j, 0]
            ney = face_normal_i[i + 1, j, 1]

            ws = w_south_j[i, j]
            wn = w_north_j[i, j]
            phi_s = ws * phi[i + 1, j] + wn * phi[i + 1, j + 1]
            nsx = face_normal_j[i, j, 0]
            nsy = face_normal_j[i, j, 1]

            ws = w_south_j[i, j + 1]
            wn = w_north_j[i, j + 1]
            phi_n = ws * phi[i + 1, j + 1] + wn * phi[i + 1, j + 2]
            nnx = face_normal_j[i, j + 1, 0]
            nny = face_normal_j[i, j + 1, 1]

            flux_x = nex * phi_e - nwx * phi_w + nnx * phi_n - nsx * phi_s
            flux_y = ney * phi_e - nwy * phi_w + nny * phi_n - nsy * phi_s
            grad[i, j, 0] = flux_x / volume[i, j]
            grad[i, j, 1] = flux_y / volume[i, j]
    return grad
