import numpy as np
from pathlib import Path

log = Path(r"C:\Users\altoa\Desktop\CHT\Project\Project 1\CFD\Project 1\komega_channel_turb1\postproc\komega_channel_friction_log.txt")
data = np.loadtxt(log, skiprows=1)

# columns: step time f_darcy f_dpdx tau_avg u_bulk Re  (per your log prints)
f = data[:,2]
f_dpdx = data[:,3]
Re = data[:,6]

# average last N points to reduce noise
N = 200
f_end = f[-N:].mean()
f_dpdx_end = f_dpdx[-N:].mean()
Re_end = Re[-N:].mean()

print("Re ~", Re_end)
print("f_D (wall) ~", f_end)
print("f_D (dpdx) ~", f_dpdx_end)
print("rel diff =", abs(f_end - f_dpdx_end)/f_end)

