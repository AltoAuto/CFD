import numpy as np

def calculate_mesh(height, nu, Re_target, target_yplus=0.8, max_ratio=1.12, cfl_target=0.5):
    """Deterministically calculates mesh and time parameters for wall-resolved CFD."""
    Dh = 4.0 * height
    U_bulk = (Re_target * nu) / Dh

    f_darcy = 0.316 * (Re_target ** -0.25)
    u_tau = U_bulk * np.sqrt(f_darcy / 8.0)

    y1 = (target_yplus * nu) / u_tau
    dy1 = 2.0 * y1

    # Calculate required cell count
    N_exact = np.log(1.0 + (height * (max_ratio - 1.0)) / dy1) / np.log(max_ratio)
    count_j = int(np.ceil(N_exact))

    # Robust Bisection Search for exact ratio_j
    target_sum = height / dy1
    r_low = 1.000001
    r_high = max_ratio + 0.1  # Safe upper bound

    for _ in range(100):
        r_mid = 0.5 * (r_low + r_high)
        # Calculate current geometric sum
        current_sum = (r_mid ** count_j - 1.0) / (r_mid - 1.0)

        if current_sum > target_sum:
            r_high = r_mid
        else:
            r_low = r_mid

    ratio_j = r_mid
    dt_max = (cfl_target * dy1) / U_bulk

    return {
        "count_j": count_j,
        "ratio_j": round(ratio_j, 5),
        "dt_max": round(dt_max, 6)
    }