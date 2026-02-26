import numpy as np
from pathlib import Path

LOG = Path(r"C:\Users\altoa\Desktop\CHT\Project\Project 1\CFD\Runs\Fine_mesh\postproc\komega_channel_friction_log.txt")

def f_darcy_haaland(Re, eD=0.0):
    """Darcy friction factor using Haaland; smooth if eD=0."""
    Re = float(abs(Re))
    if Re <= 0:
        return np.nan
    if Re < 2300:
        return 64.0 / Re
    inv_sqrt_f = -1.8 * np.log10((eD / 3.7) ** 1.11 + 6.9 / Re)
    return 1.0 / (inv_sqrt_f ** 2)

# Read whitespace-delimited log with header
d = np.genfromtxt(LOG, names=True)

# Pull arrays
step = d["step"].astype(int)
time = d["time"]
f_wall = d["f_darcy"]
f_dpdx = d["f_darcy_dpdx"]
tau = d["tau_avg"]
u_bulk = d["u_bulk"]
Re = d["reynolds"]

# Tail-average (last N points)
N = 200
N = min(N, len(step))
sl = slice(-N, None)

Re_end = np.mean(np.abs(Re[sl]))
f_wall_end = np.mean(np.abs(f_wall[sl]))
f_dpdx_end = np.mean(np.abs(f_dpdx[sl]))

# smooth wall
eD = 0.0
f_ref = f_darcy_haaland(Re_end, eD=eD)

# Relative difference between the two computed methods
rel_diff = np.nan if f_wall_end == 0 else abs(f_wall_end - f_dpdx_end) / f_wall_end

# Print last line + tail summary
i = 1
print(f"Last step={step[i]}  time={time[i]:.6e}  Re={Re[i]:.6g}  u_bulk={u_bulk[i]:.6g}")
print(f"  f_wall={f_wall[i]:.6g}  f_dpdx={f_dpdx[i]:.6g}  tau_avg={tau[i]:.6g}")
print("-" * 50)
print(f"Tail avg over last {N} points:")
print(f"  Re_end      = {Re_end:.6g}")
print(f"  f_wall_end  = {f_wall_end:.6g}")
print(f"  f_dpdx_end  = {f_dpdx_end:.6g}")
print(f"  rel_diff    = {rel_diff:.6g}")
print(f"  f_ref(Haaland, e/D={eD}) = {f_ref:.6g}")

diff = (f_wall_end - f_ref)/f_ref
print(f"  percent:{diff*100:.3g} | target = 5-10%")