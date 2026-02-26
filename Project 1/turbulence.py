import numpy as np
import bc

def _k_omega_defaults():
    """
    returning k_omega parameter. values are base on lecture slide material
    """
    params = {
        "alpha": 5.0 / 9.0,
        "beta": 3.0 / 40.0,
        "beta_star": 0.09,

        "sigma_k": 0.5,
        "sigma_omega": 0.5,

        # guard lines
        "k_min": 1.0e-15,
        "omega_min": 1.0e-6,

        # Eddy viscosity definition: nu_t = nu_t_coeff * k / omega
        "nu_t_coeff": 1,

        # Near wall condition
        "omega_wall_coeff": 85.0,
    }

    return params

def initialize(fields, mesh, cfg, bc_cfg=None):
    """Initialize turbulence fields or zero them when disabled."""
    if not cfg.get("enabled", False) or cfg.get("model") in (None, "none"):
        if "nu_t" not in fields:
            count_i = mesh["count_i"].item()
            count_j = mesh["count_j"].item()
            fields["nu_t"] = np.zeros((count_i + 2, count_j + 2), dtype=float)
        else:
            fields["nu_t"][:, :] = 0.0
        return fields

    model = cfg.get("model")
    if model not in ("k_omega",):
        raise NotImplementedError(f"Unknown turbulence model: {cfg.get('model')}")

    params = _k_omega_defaults()
    if bc_cfg is not None:
        for boundary, segments in bc_cfg.items():
            if boundary == "pressure_reference" or segments is None:
                continue
            if isinstance(segments, dict):
                segments = [segments]
            for seg in segments:
                if seg.get("type") == "inlet":
                    missing = ("k" not in seg) or ("omega" not in seg)
                    if missing:
                        raise ValueError(
                            "Turbulence inlet requires 'k' and 'omega' values."
                        )

    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    shape = (count_i + 2, count_j + 2)
    interior = (slice(1, -1), slice(1, -1))

    if "k" not in fields:
        raise ValueError("Initial k is missing; set it in initial scalars.")
    if "omega" not in fields:
        raise ValueError("Initial omega is missing; set it in initial scalars.")
    if "nu_t" not in fields:
        fields["nu_t"] = np.zeros(shape, dtype=float)
    return fields


def _apply_komega_wall(fields, mesh, boundary, index_range, nu, params):
    """Apply wall-resolved k-omega BCs at the first cell center."""
    # Inputs:
    # - fields: dict with "k" and "omega" arrays shaped (count_i + 2, count_j + 2)
    #   including ghost cells.
    # - mesh: mesh dict providing geometry and boundary helpers.
    # - boundary: boundary key (e.g., "i_min", "i_max", "j_min", "j_max").
    # - index_range: optional (start, end) slice on the boundary.
    # - nu: molecular viscosity (scalar).
    # - params: dict of model constants including "omega_wall_coeff" and "omega_min".
    # Guidance: impose wall-resolved k-omega boundary values using wall distance,
    # keeping omega bounded and ensuring boundary/ghost cells are consistent.
    # Returns: None (fields are mutated in place).
    # Solver-specific requirements:
    # - Use bc._boundary_geometry(...) to get s and use |s| as wall distance.
    # - Apply values on boundary/ghost cells via bc.set_dirichlet.
    # - If you overwrite interior, use the "interior" slice from
    #   bc._boundary_indices(...) (first cell centers adjacent to the wall).
    # - Enforce omega_min from params before applying.
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    interior, ghost = bc._boundary_indices(boundary, index_range, count_i, count_j)
    _, _, s = bc._boundary_geometry(mesh, boundary, index_range)

    if np.ndim(s) > 1 and s.shape[-1] == 2:
        y = np.linalg.norm(s, axis=-1)
    else:
        y = np.abs(s)

    y_safe = np.maximum(y, 1.0e-12)

    # Calculate wall omega based on distance
    omega_wall = params["omega_wall_coeff"] * nu / (y_safe ** 2)
    omega_wall = np.maximum(omega_wall, params["omega_min"])

    # Overwrite the interior cell values adjacent to the wall
    fields["k"][interior] = params["k_min"]
    fields["omega"][interior] = omega_wall

    # Apply standard Dirichlet mirroring to the ghost cells
    bc.set_dirichlet(fields["k"], mesh, boundary, index_range, params["k_min"])
    bc.set_dirichlet(fields["omega"], mesh, boundary, index_range, omega_wall)

def apply_bcs(fields, mesh, bc_cfg, cfg, nu):
    """Apply turbulence BCs for the active model."""
    if not cfg.get("enabled", False) or cfg.get("model") in (None, "none"):
        return

    params = _k_omega_defaults()
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    range_i, range_j = bc.get_periodic_pairs(bc_cfg, count_i, count_j)
    if range_i is not None:
        bc.apply_periodic_pair(
            {"k": fields["k"], "omega": fields["omega"]}, mesh, "i", range_i
        )
    if range_j is not None:
        bc.apply_periodic_pair(
            {"k": fields["k"], "omega": fields["omega"]}, mesh, "j", range_j
        )

    for boundary, segments in bc_cfg.items():
        if boundary == "pressure_reference" or segments is None:
            continue
        if isinstance(segments, dict):
            segments = [segments]
        for seg in segments:
            bc_type = seg.get("type")
            index_range = seg.get("range")
            if bc_type == "inlet":
                bc.set_dirichlet(fields["k"], mesh, boundary, index_range, seg["k"])
                bc.set_dirichlet(
                    fields["omega"], mesh, boundary, index_range, seg["omega"]
                )
            elif bc_type == "wall":
                _apply_komega_wall(fields, mesh, boundary, index_range, nu, params)
            elif bc_type in ("outlet", "wall", "symmetry"):
                if "k" in seg:
                    bc.set_dirichlet(fields["k"], mesh, boundary, index_range, seg["k"])
                else:
                    bc.set_neumann(
                        fields["k"], mesh, boundary, index_range, seg.get("k_gradient", 0.0)
                    )
                if "omega" in seg:
                    bc.set_dirichlet(
                        fields["omega"], mesh, boundary, index_range, seg["omega"]
                    )
                else:
                    bc.set_neumann(
                        fields["omega"],
                        mesh,
                        boundary,
                        index_range,
                        seg.get("omega_gradient", 0.0),
                    )
            elif bc_type == "periodic":
                continue
            else:
                raise ValueError(f"Unknown BC type for turbulence: {bc_type}")


def eddy_viscosity(fields, cfg):
    """Compute eddy viscosity nu_t from k and omega."""
    # Inputs:
    # - fields: dict containing arrays "k", "omega", "nu_t", each shaped
    #   (count_i + 2, count_j + 2) including ghost cells.
    # - cfg: dict with turbulence settings (enabled/model).
    # Guidance: update nu_t in place using k and omega with sensible minima and
    # leave a zero field when turbulence is disabled.
    # Returns: fields["nu_t"] (array shaped (count_i + 2, count_j + 2)).
    # Solver-specific requirements:
    # - Read k and omega from fields (same ghosted shape as nu_t).
    # - Enforce k_min and omega_min from _k_omega_defaults() before division.
    # - nu_t must be non-negative and stored in fields["nu_t"].

    if not cfg.get("enabled", False) or cfg.get("model") in (None, "none"):
        fields["nu_t"][:, :] = 0.0
        return fields["nu_t"]

    params = _k_omega_defaults()
    k_min = params["k_min"]
    omega_min = params["omega_min"]
    nu_t_coeff = params["nu_t_coeff"]

    k_safe = np.maximum(fields["k"], k_min)
    omega_safe = np.maximum(fields["omega"], omega_min)

    fields["nu_t"][:, :] = nu_t_coeff * k_safe / omega_safe
    return fields["nu_t"]


def sources(fields, grad_u, grad_v):
    """Compute turbulence source terms for the active model."""
    # Inputs:
    # - fields: dict with "k", "omega", "nu_t" arrays shaped
    #   (count_i + 2, count_j + 2) including ghosts.
    # - grad_u, grad_v: cell-centered gradients shaped (count_i, count_j, 2)
    #   for interior cells only (no ghosts).
    # Guidance: compute production/dissipation source terms from velocity
    # gradients using safe minima for k/omega, returning interior arrays.
    # Returns: (source_k, source_omega), each shaped (count_i, count_j).
    # Solver-specific requirements:
    # - grad_u/grad_v are interior-only; outputs must be interior-only too.
    # - Use fields["nu_t"][1:-1, 1:-1] for production.
    # - Enforce k_min and omega_min from _k_omega_defaults() before division.
    params = _k_omega_defaults()
    k = np.maximum(fields["k"][1:-1, 1:-1], params["k_min"])
    omega = np.maximum(fields["omega"][1:-1, 1:-1], params["omega_min"])
    nu_t = fields["nu_t"][1:-1, 1:-1]
    ux = grad_u[..., 0]
    uy = grad_u[..., 1]
    vx = grad_v[..., 0]
    vy = grad_v[..., 1]
    S2 = 2.0 * ux ** 2 + 2.0 * vy ** 2 + (uy + vx) ** 2
    P_k = nu_t * S2
    gamma = params["alpha"]
    beta = params["beta"]
    beta_star = params["beta_star"]

    # --- K-EQUATION SOURCE ---
    # Sk = Pk - beta_star * k * omega
    source_k = P_k - beta_star * k * omega

    # --- OMEGA-EQUATION SOURCE (From lecture slide)---
    # Sw = gamma * (omega / k) * Pk - beta * omega^2
    #source_omega = gamma * (omega / k) * P_k - beta * (omega ** 2)

    # --- OMEGA-EQUATION SOURCE (More stable) ---
    # Sw = gamma * S^2 - beta * omega^2
    source_omega = gamma * S2 - beta * (omega ** 2)

    return source_k, source_omega

def effective_diffusivity(nu, nu_t, field="k"):
    """Return effective diffusivity for k or omega transport."""
    # Inputs:
    # - nu: molecular viscosity (scalar).
    # - nu_t: eddy viscosity array shaped (count_i + 2, count_j + 2) including
    #   ghosts.
    # - field: "k" or "omega" selects sigma_k or sigma_omega.
    # Guidance: compute the effective diffusivity for the requested transport
    # equation without mutating inputs; nu is scalar.
    # Returns: effective diffusivity array (same shape as nu_t when nu_t is array).
    # Solver-specific requirements:
    # - For "k" use sigma_k, for "omega" use sigma_omega (from _k_omega_defaults()).
    # - Returned array must be compatible with slicing [1:-1, 1:-1].

    params = _k_omega_defaults()

    if field == "k":
        sigma = params["sigma_k"]
    elif field == "omega":
        sigma = params["sigma_omega"]
    else:
        raise ValueError(f"Unsupported field '{field}' for effective_diffusivity.")

    return nu + sigma * nu_t