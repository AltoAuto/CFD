import numpy as np

import bc

# Model requirements:
# - k-omega needs k and omega fields, plus nu_t derived from them.
# - Boundary conditions for k and omega are required at inlets (Dirichlet).
# - Outlet/wall/symmetry default to zero normal gradient (resolved wall).


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
    interior, _ = bc._boundary_indices(boundary, index_range, count_i, count_j)
    _, _, s = bc._boundary_geometry(mesh, boundary, index_range)
    y = np.abs(s)
    y_safe = np.where(y > 1.0e-12, y, 1.0e-12)

    omega_min = float(params["omega_min"])
    Cw = float(params["omega_wall_coeff"])

    # omega_wall = Cw * ν / y^2
    omega_wall = Cw * nu / (y_safe**2)
    omega_wall = np.maximum(omega_wall, omega_min)

    # set frist real cell next to the wall omega
    fields["omega"][interior] = omega_wall

    # set ghost cell, omega_wall, k=0
    bc.set_dirichlet(fields["omega"], mesh, boundary, index_range, omega_wall)
    bc.set_dirichlet(fields["k"], mesh, boundary, index_range, 0.0)

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

    if "nu_t" not in fields:
        raise ValueError("fields['nu_t'] is missing.")

    # make nu_t everywhere for laminar
    if (not cfg.get("enabled", False)) or (cfg.get("model") in (None, "none")):
        fields["nu_t"][:, :] = 0.0
        return fields["nu_t"]

    # this is for turbulence
    model = cfg.get("model")
    if model not in ("k_omega",):
        raise NotImplementedError(f"Unknown turbulence model: {model}")

    if "k" not in fields or "omega" not in fields:
        raise ValueError("fields['k'] and fields['omega'] are missing.")

    # Get all the Params
    params = _k_omega_defaults()
    k_min = float(params["k_min"])
    omega_min = float(params["omega_min"])

   # fields["k"][1:-1, 1:-1] = np.maximum(fields["k"][1:-1, 1:-1], 0.0)
   # fields["omega"][1:-1, 1:-1] = np.maximum(fields["omega"][1:-1, 1:-1], omega_min)

    nu_t_coeff = float(params.get("nu_t_coeff", 1.0))
    k = fields["k"]
    omega = fields["omega"]
    nu_t = fields["nu_t"]

    # make sure k and omega never goes negative or zero
    k_safe = np.maximum(k, k_min)
    omega_safe = np.maximum(omega, omega_min)

    # nu_t = C * k / omega
    nu_t[:, :] = nu_t_coeff * (k_safe / omega_safe)

    # Enforce non-negative and finite
    np.maximum(nu_t, 0.0, out=nu_t)
    nu_t[~np.isfinite(nu_t)] = 0.0

    return nu_t

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
    params= _k_omega_defaults()
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    beta_star = float(params["beta_star"])
    k_min = float(params["k_min"])
    omega_min = float(params["omega_min"])

    # getting rid of ghost cells
    nu_t = fields["nu_t"][1:-1, 1:-1]
    k = np.maximum(fields["k"][1:-1, 1:-1], k_min)
    omega = np.maximum(fields["omega"][1:-1, 1:-1], omega_min)

    du_dx = grad_u[:, :, 0]
    du_dy = grad_u[:, :, 1]
    dv_dx = grad_v[:, :, 0]
    dv_dy = grad_v[:, :, 1]

    # Strain-rate components
    S11 = du_dx
    S22 = dv_dy
    S12 = 0.5 * (du_dy + dv_dx)

    # Invariant SijSij and production Pk = 2*nu_t*SijSij
    SijSij = S11**2 + S22**2 + 2.0 * (S12**2)
    Pk = 2.0 * nu_t * SijSij
    Pk = np.maximum(Pk, 0.0)

    # k equation source: Pk - beta* k*omega
    source_k = Pk - beta_star * k * omega

    # omega equation source: alpha*(omega/k)*Pk - beta*omega^2
    source_omega = alpha * (omega / k) * Pk - beta * (omega**2)

    # Defensive cleanup
    source_k[~np.isfinite(source_k)] = 0.0
    source_omega[~np.isfinite(source_omega)] = 0.0
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
        sigma = float(params["sigma_k"])
    elif field in ("omega", "w"):
        sigma = float(params["sigma_omega"])
    else:
        raise ValueError(f"Unknown effective diffusivity field name{field}")

    nu_eff = nu + (1/sigma)* nu_t

    # guard-lines, nu_eff always > nu, get rid of Nan values
    nu_eff = np.maximum(nu_eff, nu)
    nu_eff[~np.isfinite(nu_eff)] = nu

    return nu_eff
