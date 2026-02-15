from pathlib import Path

import numpy as np

from mesh import generate_rect_mesh, make_mesh

CASE_NAME = "rectangular_template"

CASE_DIR = Path(__file__).resolve().parent
MESH_DIR = CASE_DIR / "mesh"
STATES_DIR = CASE_DIR / "states"
POST_DIR = CASE_DIR / "postproc"
MESH_PATH = MESH_DIR / f"{CASE_NAME}.npz"

MESH = {
    "generator": generate_rect_mesh,
    "params": {
        "length_i": 1.0,
        "length_j": 1.0,
        "count_i": 32,
        "count_j": 32,
        "ratio_i": 1.0,
        "ratio_j": 1.0,
    },
    "path": MESH_PATH,
}

TIME = {
    "cfl": 0.5,
    "cfl_diff": 0.5,
    "dt_max": 1.0e-3,
    "dt_min": 0.0,
    "max_steps": 10000,
}

DPDX = -1e-1

PHYSICS = {
    "nu": 1.0e-3,
    "pressure_gradient": (DPDX, 0.0),
}

SOLVER = {
    "scheme": "simple",
    "use_numba": False,
    "momentum": {
        "tol": 1.0e-8,
        "maxiter": 200,
    },
    "pressure": {
        "tol": 1.0e-8,
        "maxiter": 200,
    },
    "piso": {
        "n_correctors": 3,
        "corrector_schedule": None,
    },
    "rhie_chow": True,
    "simple": {
        "max_iter": 10,
        "continuity_tol": 1.0e-4,
        "relax_u": 0.7,
        "relax_v": 0.7,
        "relax_p": 0.3,
    },
}

RESTART = {
    "enabled": False,
    "step": None,
}

TURBULENCE = {
    "enabled": False,
    "model": "none",
    "params": {},
}

POST = {
    "plot_interval": 200,
    "save_interval": 100,
    "fields": ("u", "v", "p"),
    "residual_log_interval": 1,
    "friction_log_interval": 1,
    "friction": {
        "enabled": False,
        "boundary": "j_min",
        "range": None,
        "u_ref": None,
        "rho": 1.0,
        "use_nu_t": False,
        "flow_axis": "i",
        "hydraulic_diameter": None,
    },
}

BOUNDARY = {
    "i_min": {"u": 0.0, "v": 0.0, "p": None},
    "i_max": {"u": 0.0, "v": 0.0, "p": None},
    "j_min": {"u": 0.0, "v": 0.0, "p": None},
    "j_max": {"u": 0.0, "v": 0.0, "p": None},
    "pressure_reference": {"cell": (0, 0), "value": 0.0},
}

INITIAL = {
    "u": 0.0,
    "v": 0.0,
    "p": 0.0,
    "scalars": {},
}


def initial_conditions(mesh):
    """Construct initial field arrays from defaults."""
    shape = mesh["cell_center"].shape[:2]
    u = np.full(shape, INITIAL["u"], dtype=float)
    v = np.full(shape, INITIAL["v"], dtype=float)
    p = np.full(shape, INITIAL["p"], dtype=float)
    return {"u": u, "v": v, "p": p}


CONFIG = {
    "case_name": CASE_NAME,
    "paths": {
        "case_dir": CASE_DIR,
        "mesh_dir": MESH_DIR,
        "states_dir": STATES_DIR,
        "post_dir": POST_DIR,
    },
    "mesh": MESH,
    "time": TIME,
    "physics": PHYSICS,
    "solver": SOLVER,
    "restart": RESTART,
    "turbulence": TURBULENCE,
    "post": POST,
    "boundary": BOUNDARY,
    "initial": INITIAL,
}


def run_case():
    """Generate the mesh for this case."""
    for path in (MESH_DIR, STATES_DIR, POST_DIR):
        path.mkdir(parents=True, exist_ok=True)
    mesh = make_mesh(MESH)
    return mesh


def run_solver():
    """Placeholder for solver entry point in this template."""
    raise NotImplementedError("Solver entry point not wired yet.")


if __name__ == "__main__":
    run_case()
