# 2D Incompressible FV Solver

## Features
- Collocated finite-volume formulation with Rhie–Chow interpolation.
- SIMPLE-style pressure–velocity coupling (pseudo-time stepping).
- Explicit advection (1st-order upwind), implicit diffusion.
- Curvilinear grids via node-based geometry + precomputed metrics.
- Ghost-cell boundary treatment (inlet, outlet, wall, symmetry, periodic).
- k–omega turbulence model (eddy viscosity) with optional p* = p + 2/3 k.
- Adaptive time stepping using a global CFL constraint.
- Post-processing helpers (mesh plot, fields, convergence, y+, line probes, surface forces).

## Directory Layout
Each case is a self-contained folder:
```
case/
  config.py
  mesh/        # one mesh .npz + mesh plot
  states/      # saved fields_XXXXXX.npz (includes ghost cells)
  postproc/    # convergence, field plots, comparisons
```
Only one mesh and one residual log are allowed per case directory.

## Dependencies
- Python 3.x
- numpy
- scipy
- matplotlib
- pyamg (assumed installed; used for pressure solve)

## Quick Start
From a case directory:
```
python config.py
```
Example cases:
- `test/` (channel flow)
- `test2/` (Couette flow)
- `half_cylinder/`
- `half_cylinder_turbulent/`

If running headless:
```
MPLBACKEND=Agg python config.py
```

## Config Structure (config.py)
Each case builds a `CONFIG` dict and calls `solver.run_case(CONFIG)` (returns `fields, step, time`). Typical keys:
```
CONFIG = {
  "mesh": MESH,
  "time": TIME,
  "physics": PHYSICS,
  "solver": SOLVER,
  "boundary": BOUNDARY,
  "turbulence": TURBULENCE,
  "restart": RESTART,
  "post": POST,
  "initial": INITIAL,
  "paths": PATHS,
}
```

### Time / Restart
- `TIME["max_steps"]` defines the number of outer steps.
- If `TIME["cfl"]` is set, the solver computes `dt` each step from the minimum cell-wise CFL:
  - `dt = cfl * V / (|F_w| + |F_e| + |F_s| + |F_n|)`
  - Optional clamps: `TIME["dt_min"]`, `TIME["dt_max"]`.
- Optional diffusive CFL: `TIME["cfl_diff"]` limits `dt` using `min(Δl^2 / nu_eff)`.
- Optional schedule: `TIME["cfl_diff_schedule"] = [(cfl_diff, step), ...]` linearly ramps the diffusive CFL.

### Logging Intervals
Control logging frequency:
```
POST["residual_log_interval"] = 10
POST["friction_log_interval"] = 10
```

### Friction Coefficient Logging
Enable logging/plotting of Darcy friction factor:
```
POST["friction"] = {
  "enabled": True,
  "boundary": "j_min",
  "range": None,
  "u_ref": 1.0,
  "rho": 1.0,
  "use_nu_t": False,
  "flow_axis": "i",
  "hydraulic_diameter": None,
}
```
- Logs Darcy friction factor `f = 4 * C_f` and `f_dpdx = 4 * C_f,dpdx`.
- If `u_ref` is `None`, it is computed as the mean velocity over the full interior
  domain for the selected `flow_axis`.
- If `hydraulic_diameter` and `pressure_gradient` are available, `cf_dpdx` is logged using
  `tau_w = -(dp/dx) * D_h / 4` (or `dp/dy` for `flow_axis="j"`).
- Logs to `*_friction_log.txt` and plots after the run.
- Optional schedule: `TIME["cfl_schedule"] = [(cfl, step), ...]` linearly ramps between steps.
- Optional schedule: `TIME["dt_max_schedule"] = [(dt_max, step), ...]` linearly ramps the max dt.
- If `TIME["cfl"]` is omitted, the solver uses fixed `TIME["dt"]`.
- If `RESTART["enabled"] = True`, the solver continues for *another* `max_steps`.
- Restart state is resolved via `states/fields_XXXXXX.npz` using `RESTART["step"]`.

### Boundary Conditions
Supported types: `inlet`, `outlet`, `wall`, `symmetry`, `periodic`.
- BCs are applied to boundary segments with optional `range=(start,end)` in index space.
- Periodic must be specified on both sides of a pair (i_min+i_max or j_min+j_max).
- If a periodic boundary also specifies `pressure`, velocity/scalars remain periodic but pressure is clamped.
- Pressure reference is used if no pressure Dirichlet outlet is specified.

### Pressure Gradient Forcing
You can apply a uniform driving pressure gradient via:
```
PHYSICS["pressure_gradient"] = (dpdx, dpdy)
```
This adds a constant body force `(-dpdx, -dpdy)` to the momentum equations (rho=1).

### Pressure AMG Reuse
AMG hierarchy is reused automatically for the pressure solve to reduce setup cost.

### SIMPLE vs PISO
Select the algorithm via:
```
SOLVER["scheme"] = "simple"  # or "piso"
```
For PISO, set the number of pressure corrections per step:
```
SOLVER["piso"]["n_correctors"] = 3
```
You can also schedule the number of correctors:
```
SOLVER["piso"]["corrector_schedule"] = [(3, 0), (2, 5000), (1, 20000)]
```

### Numba Acceleration
Enable Numba-accelerated advection/gradient kernels:
```
SOLVER["use_numba"] = True
```

### Turbulence (k–omega)
Enable with:
```
TURBULENCE = {"enabled": True, "model": "k_omega", "params": {...}}
```
Notes:
- Requires `k` and `omega` at inlets (Dirichlet).
- Outlet/wall/symmetry default to zero-gradient for k/omega (unless overridden).
- The isotropic term `2/3 k` is applied as an explicit pressure-like source in momentum.
- Optional update interval: `TURBULENCE["update_interval"] = N` (default 1).

## Mesh Generators (mesh.py)
- `generate_rect_mesh` (rectangular)
- `generate_ramp_mesh` (linear contraction/expansion)
- `generate_annulus_mesh` (concentric annulus, graded)
- `precompute_from_nodes` works for any curvilinear node array.

## Post-Processing (post_processing.py)
Core functions:
- `plot_mesh(mesh_path_or_dir)`
- `plot_field(state, mesh, field="u|v|p|speed")`
- `plot_line(state, mesh, field, start, end)`
- `plot_yplus(state, mesh, boundary, ...)`
- `plot_convergence(log_path_or_dir)` (also plots momentum residuals)
- `surface_force(state, mesh, boundary, ...)` (pressure + shear)

## Outputs
- `states/fields_XXXXXX.npz` includes: `u`, `v`, `p`, and turbulence fields (`k`, `omega`, `nu_t`) if enabled.
- `postproc/*_residual_log.txt` contains continuity + momentum residuals.
- Plots are saved in `postproc/`.
 - `POST["print_interval"]` controls console output frequency (0 disables).
 - `POST["timing"]` enables per-step timing logs and plots.

## Notes / Limitations
- Advection is first-order upwind (stable, diffusive).
- Periodic turbulence BCs are supported; mixed periodic/non-periodic on same boundary is not.

## Extension Points
Add new features in these modules:
- New mesh generator: `mesh.py`
- New turbulence model: `turbulence.py`
- New BC types: `bc.py`
- New post-processing tools: `post_processing.py`

Keep the interfaces consistent so cases remain plug-and-play.

## Developer Notes
- The solver is intentionally minimal; prefer readable loops over heavy abstraction.
- All fields are cell-centered with ghost cells; face data comes from precomputed geometry.
- Periodic boundaries must be paired; do not mix periodic with other BC types on a boundary.
- Residual logs are appended per outer step; restarting truncates logs to the restart step.
- If you change log format, update `plot_convergence` accordingly.
- For curvilinear grids, normals and face sizes are computed from node geometry.

## Solver Function Guide (Key Routines)
- `solver.run_case(CONFIG)`: main driver; handles restart, loops over time/outer iterations, logging, and saving. Returns `(fields, step, time)`.
- `solver.apply_boundary_conditions(...)`: applies all BCs via ghost-cell updates.
- `solver.compute_advective_rhs(...)`: 1st-order upwind advection for any scalar field.
- `solver.solve_momentum(...)`: implicit diffusion + transient solve for u and v.
- `solver.solve_pressure(...)`: pressure-correction Poisson solve (SIMPLE).
- `solver.correct_velocity(...)`: applies pressure correction and under-relaxation.
- `turbulence.update_turbulence(...)`: advances k and omega and updates `nu_t`.
- `mesh.precompute_from_nodes(...)`: computes geometry (cell centers, face centers, normals, sizes).
- `post_processing.plot_*`: plotting utilities; `surface_force` computes force on a boundary.

## Math Outline (Each Outer Iteration)
0) **Adaptive time step (optional)**
   - If CFL control is enabled, compute `dt` from the minimum cell CFL before the SIMPLE loop.
1) **Apply BCs** (ghost cells)
   - Velocity, pressure, and turbulence scalars are enforced on boundary segments.

2) **Advection (explicit)**
   - For any scalar φ: `∂(φV)/∂t + ∇·(uφ) = ...`
   - Upwind face values use local mass flux sign.

3) **Momentum predictor (implicit diffusion)**
   - Solve for u* and v*:
     - `(V/Δt) u* - ∇·(ν_eff ∇u*) = (V/Δt) u^n - ∇p* + advection`
   - `ν_eff = ν + ν_t`, `p* = p + 2/3 k` (if turbulence enabled).

4) **Pressure correction (SIMPLE)**
   - Compute face mass fluxes with Rhie–Chow:
     - `F_f = (u_f · S_f) - (|S_f|^2 / a_f) (p_N - p_P)`
   - Solve Poisson for pressure correction:
     - `∇·( (|S|^2 / a_f) ∇p' ) = -∇·F*`
   - If no Dirichlet pressure, fix one reference cell.

5) **Velocity correction**
   - `u = u* - (V/a_p) ∇p'`
   - Apply under-relaxation.

6) **Turbulence update (k–ω)**
   - `k` and `ω` transport:
     - `∂k/∂t + ∇·(uk) = ∇·( (ν+ν_t/σ_k) ∇k ) + P_k - β* k ω`
     - `∂ω/∂t + ∇·(uω) = ∇·( (ν+ν_t/σ_ω) ∇ω ) + α P_k / k - β ω^2`
- `ν_t = k / ω` (with model coefficients and clipping)

7) **Convergence check**
   - Continuity L2 and momentum L2 are logged each outer step.

This loop repeats until reaching `max_steps`.

## Credits & Acknowledgments
This project was developed at the **University of Minnesota**.

* **Core Solver & Post-Processing Architecture:** Provided by the UMN Department of Mechanical Engineering.
* **Implementation of turbulence Logic:** Developed by Aiden Wang.
