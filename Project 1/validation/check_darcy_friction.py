import numpy as np
from pathlib import Path

LOG = Path(r"C:\Users\altoa\Desktop\CHT\Project\Project 1\CFD\Project 1\komega_channel_turb1\postproc\komega_channel_friction_log.txt")

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

# Relative difference between the two computed methods
rel_diff = np.nan if f_wall_end == 0 else abs(f_wall_end - f_dpdx_end) / f_wall_end

# Print last line + tail summary
i = -1
print(f"Last step={step[i]}  time={time[i]:.6e}  Re={Re[i]:.6g}  u_bulk={u_bulk[i]:.6g}")
print(f"  f_wall={f_wall[i]:.6g}  f_dpdx={f_dpdx[i]:.6g}  tau_avg={tau[i]:.6g}")
print("-" * 50)
print(f"Tail avg over last {N} points:")
print(f"  Re_end      = {Re_end:.6g}")
print(f"  f_wall_end  = {f_wall_end:.6g}")
print(f"  f_dpdx_end  = {f_dpdx_end:.6g}")
print(f"  rel_diff    = {rel_diff:.6g}")
