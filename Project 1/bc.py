import numpy as np


def _parse_index_range(index_range, max_index):
    """Normalize and validate an index range on a boundary."""
    if index_range is None:
        return 0, max_index
    if len(index_range) != 2:
        raise ValueError("index_range must be (start, end)")
    start, end = index_range
    if start < 0 or end < start or end > max_index:
        raise ValueError("index_range out of bounds")
    return start, end


def _boundary_indices(boundary, index_range, count_i, count_j):
    """Return interior and ghost index slices for a boundary segment."""
    if boundary in ("i_min", "i_max"):
        start, end = _parse_index_range(index_range, count_j - 1)
        j_slice = slice(start + 1, end + 2)
        i_int = 1 if boundary == "i_min" else count_i
        i_ghost = 0 if boundary == "i_min" else count_i + 1
        interior = (i_int, j_slice)
        ghost = (i_ghost, j_slice)
    elif boundary in ("j_min", "j_max"):
        start, end = _parse_index_range(index_range, count_i - 1)
        i_slice = slice(start + 1, end + 2)
        j_int = 1 if boundary == "j_min" else count_j
        j_ghost = 0 if boundary == "j_min" else count_j + 1
        interior = (i_slice, j_int)
        ghost = (i_slice, j_ghost)
    else:
        raise ValueError(f"Unknown boundary: {boundary}")

    return interior, ghost


def _normalize_segments(segments):
    """Normalize BC segments input into a list of segment dicts."""
    if segments is None:
        return []
    if isinstance(segments, dict):
        return [segments]
    return segments


def _periodic_segment(segments):
    """Extract the periodic segment or return None if not periodic."""
    if not segments:
        return None
    types = {seg.get("type") for seg in segments}
    if "periodic" not in types:
        return None
    if types != {"periodic"}:
        raise ValueError("Periodic boundary cannot be mixed with other BC types.")
    if len(segments) != 1:
        raise ValueError("Periodic boundary must use a single segment.")
    return segments[0]


def _match_periodic_range(seg_a, seg_b, max_index):
    """Validate periodic ranges and return the matching index range."""
    start_a, end_a = _parse_index_range(seg_a.get("range"), max_index)
    start_b, end_b = _parse_index_range(seg_b.get("range"), max_index)
    if (start_a, end_a) != (start_b, end_b):
        raise ValueError("Periodic boundary ranges must match on both sides.")
    if start_a != 0 or end_a != max_index:
        raise ValueError("Periodic boundaries must span the full boundary range.")
    return start_a, end_a


def get_periodic_pairs(bc_cfg, count_i, count_j):
    """Return periodic index ranges for i and j boundaries."""
    seg_i_min = _periodic_segment(_normalize_segments(bc_cfg.get("i_min")))
    seg_i_max = _periodic_segment(_normalize_segments(bc_cfg.get("i_max")))
    seg_j_min = _periodic_segment(_normalize_segments(bc_cfg.get("j_min")))
    seg_j_max = _periodic_segment(_normalize_segments(bc_cfg.get("j_max")))

    if (seg_i_min is None) != (seg_i_max is None):
        raise ValueError("Periodic BC must be specified on both i_min and i_max.")
    if (seg_j_min is None) != (seg_j_max is None):
        raise ValueError("Periodic BC must be specified on both j_min and j_max.")

    range_i = None
    range_j = None
    if seg_i_min is not None:
        range_i = _match_periodic_range(seg_i_min, seg_i_max, count_j - 1)
    if seg_j_min is not None:
        range_j = _match_periodic_range(seg_j_min, seg_j_max, count_i - 1)

    return range_i, range_j


def apply_periodic_pair(fields, mesh, axis, index_range):
    """Copy interior values across paired periodic boundaries."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    if axis == "i":
        start, end = _parse_index_range(index_range, count_j - 1)
        j_slice = slice(start + 1, end + 2)
        left = 1
        right = count_i
        left_ghost = 0
        right_ghost = count_i + 1
        for arr in fields.values():
            arr[left_ghost, j_slice] = arr[right, j_slice]
            arr[right_ghost, j_slice] = arr[left, j_slice]
    elif axis == "j":
        start, end = _parse_index_range(index_range, count_i - 1)
        i_slice = slice(start + 1, end + 2)
        bottom = 1
        top = count_j
        bottom_ghost = 0
        top_ghost = count_j + 1
        for arr in fields.values():
            arr[i_slice, bottom_ghost] = arr[i_slice, top]
            arr[i_slice, top_ghost] = arr[i_slice, bottom]
    else:
        raise ValueError("axis must be 'i' or 'j'")


def _expand_value(value, shape):
    """Broadcast a scalar or validate array-valued BC input."""
    if np.isscalar(value):
        return np.full(shape, value, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"value has shape {arr.shape}, expected {shape}")
    return arr


def _boundary_geometry(mesh, boundary, index_range):
    """Return face centers, unit normals, and wall-normal distances."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    node = mesh["node"]
    cell_center = mesh["cell_center"]

    if boundary in ("i_min", "i_max"):
        start, end = _parse_index_range(index_range, count_j - 1)
        j0 = slice(start, end + 1)
        j1 = slice(start + 1, end + 2)
        if boundary == "i_min":
            face_center = mesh["face_center"][0, j0, 0, :]
            face_normal = mesh["face_normal"][0, j0, 0, :]
            cell = cell_center[0, j0, :]
        else:
            i_face = count_i
            n0 = node[i_face, j0, :]
            n1 = node[i_face, j1, :]
            face_center = 0.5 * (n0 + n1)
            t = n1 - n0
            face_normal = np.stack((t[..., 1], -t[..., 0]), axis=-1)

            i_cell = count_i - 1
            e0 = node[i_cell + 1, j0, :] - node[i_cell, j0, :]
            e1 = node[i_cell + 1, j1, :] - node[i_cell, j1, :]
            s_i = 0.5 * (e0 + e1)
            dot = np.sum(face_normal * s_i, axis=-1)
            face_normal *= np.where(dot < 0.0, -1.0, 1.0)[..., None]
            cell = cell_center[i_cell, j0, :]
    elif boundary in ("j_min", "j_max"):
        start, end = _parse_index_range(index_range, count_i - 1)
        i0 = slice(start, end + 1)
        i1 = slice(start + 1, end + 2)
        if boundary == "j_min":
            face_center = mesh["face_center"][i0, 0, 1, :]
            face_normal = mesh["face_normal"][i0, 0, 1, :]
            cell = cell_center[i0, 0, :]
        else:
            j_face = count_j
            n0 = node[i0, j_face, :]
            n1 = node[i1, j_face, :]
            face_center = 0.5 * (n0 + n1)
            t = n1 - n0
            face_normal = np.stack((-t[..., 1], t[..., 0]), axis=-1)

            j_cell = count_j - 1
            e0 = node[i0, j_cell + 1, :] - node[i0, j_cell, :]
            e1 = node[i1, j_cell + 1, :] - node[i1, j_cell, :]
            s_j = 0.5 * (e0 + e1)
            dot = np.sum(face_normal * s_j, axis=-1)
            face_normal *= np.where(dot < 0.0, -1.0, 1.0)[..., None]
            cell = cell_center[i0, j_cell, :]
    else:
        raise ValueError(f"Unknown boundary: {boundary}")

    normal_mag = np.hypot(face_normal[..., 0], face_normal[..., 1])
    n_hat = face_normal / normal_mag[..., None]
    s = np.sum((cell - face_center) * n_hat, axis=-1)
    return face_center, n_hat, s


def set_dirichlet(phi, mesh, boundary, index_range, value):
    """Apply a Dirichlet value via ghost-cell mirroring."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    interior, ghost = _boundary_indices(boundary, index_range, count_i, count_j)
    phi_p = phi[interior]
    phi_bc = _expand_value(value, phi_p.shape)
    phi[ghost] = 2.0 * phi_bc - phi_p


def set_neumann(phi, mesh, boundary, index_range, gradient):
    """Apply a Neumann gradient using ghost cells and face normals."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    interior, ghost = _boundary_indices(boundary, index_range, count_i, count_j)
    phi_p = phi[interior]
    grad = _expand_value(gradient, phi_p.shape)
    _, _, s = _boundary_geometry(mesh, boundary, index_range)
    phi[ghost] = phi_p - 2.0 * s * grad


def apply_inlet(fields, mesh, boundary, index_range, velocity, pressure_gradient=0.0):
    """Apply inlet velocity and optional pressure gradient."""
    u_in, v_in = velocity
    set_dirichlet(fields["u"], mesh, boundary, index_range, u_in)
    set_dirichlet(fields["v"], mesh, boundary, index_range, v_in)
    set_neumann(fields["p"], mesh, boundary, index_range, pressure_gradient)


def apply_outlet(fields, mesh, boundary, index_range, pressure, velocity_gradient=(0.0, 0.0)):
    """Apply outlet pressure and optional velocity gradients."""
    du_dn, dv_dn = velocity_gradient
    set_neumann(fields["u"], mesh, boundary, index_range, du_dn)
    set_neumann(fields["v"], mesh, boundary, index_range, dv_dn)
    set_dirichlet(fields["p"], mesh, boundary, index_range, pressure)


def apply_wall(fields, mesh, boundary, index_range, velocity=(0.0, 0.0), pressure_gradient=0.0):
    """Apply wall velocity and zero-gradient pressure by default."""
    u_wall, v_wall = velocity
    set_dirichlet(fields["u"], mesh, boundary, index_range, u_wall)
    set_dirichlet(fields["v"], mesh, boundary, index_range, v_wall)
    set_neumann(fields["p"], mesh, boundary, index_range, pressure_gradient)


def apply_symmetry(fields, mesh, boundary, index_range):
    """Apply symmetry by reflecting normal velocity and zero-gradient pressure."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    interior, ghost = _boundary_indices(boundary, index_range, count_i, count_j)
    _, n_hat, _ = _boundary_geometry(mesh, boundary, index_range)

    u_p = fields["u"][interior]
    v_p = fields["v"][interior]
    vel = np.stack((u_p, v_p), axis=-1)
    u_n = np.sum(vel * n_hat, axis=-1)
    vel_ghost = vel - 2.0 * u_n[..., None] * n_hat

    fields["u"][ghost] = vel_ghost[..., 0]
    fields["v"][ghost] = vel_ghost[..., 1]
    set_neumann(fields["p"], mesh, boundary, index_range, 0.0)


def apply_boundary_conditions(fields, mesh, bc_cfg):
    """Apply all configured velocity/pressure boundary conditions."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    range_i, range_j = get_periodic_pairs(bc_cfg, count_i, count_j)
    periodic_boundaries = set()
    periodic_pressure = {"i": False, "j": False}
    if range_i is not None:
        seg_i_min = _normalize_segments(bc_cfg.get("i_min"))
        seg_i_max = _normalize_segments(bc_cfg.get("i_max"))
        periodic_pressure["i"] = any("pressure" in seg for seg in seg_i_min + seg_i_max)
    if range_j is not None:
        seg_j_min = _normalize_segments(bc_cfg.get("j_min"))
        seg_j_max = _normalize_segments(bc_cfg.get("j_max"))
        periodic_pressure["j"] = any("pressure" in seg for seg in seg_j_min + seg_j_max)
    if range_i is not None:
        periodic_fields = fields if not periodic_pressure["i"] else {
            key: value for key, value in fields.items() if key != "p"
        }
        apply_periodic_pair(periodic_fields, mesh, "i", range_i)
        periodic_boundaries.update(("i_min", "i_max"))
    if range_j is not None:
        periodic_fields = fields if not periodic_pressure["j"] else {
            key: value for key, value in fields.items() if key != "p"
        }
        apply_periodic_pair(periodic_fields, mesh, "j", range_j)
        periodic_boundaries.update(("j_min", "j_max"))

    for boundary, segments in bc_cfg.items():
        if boundary == "pressure_reference":
            continue
        if boundary in periodic_boundaries:
            segments = _normalize_segments(segments)
            for seg in segments:
                if "pressure" in seg:
                    set_dirichlet(
                        fields["p"],
                        mesh,
                        boundary,
                        seg.get("range"),
                        seg["pressure"],
                    )
                elif "pressure_gradient" in seg:
                    set_neumann(
                        fields["p"],
                        mesh,
                        boundary,
                        seg.get("range"),
                        seg["pressure_gradient"],
                    )
            continue
        if segments is None:
            continue
        segments = _normalize_segments(segments)
        for seg in segments:
            bc_type = seg["type"]
            index_range = seg.get("range")
            if bc_type == "inlet":
                apply_inlet(
                    fields,
                    mesh,
                    boundary,
                    index_range,
                    velocity=seg["velocity"],
                    pressure_gradient=seg.get("pressure_gradient", 0.0),
                )
            elif bc_type == "outlet":
                apply_outlet(
                    fields,
                    mesh,
                    boundary,
                    index_range,
                    pressure=seg["pressure"],
                    velocity_gradient=seg.get("velocity_gradient", (0.0, 0.0)),
                )
            elif bc_type == "wall":
                apply_wall(
                    fields,
                    mesh,
                    boundary,
                    index_range,
                    velocity=seg.get("velocity", (0.0, 0.0)),
                    pressure_gradient=seg.get("pressure_gradient", 0.0),
                )
            elif bc_type == "symmetry":
                apply_symmetry(fields, mesh, boundary, index_range)
            else:
                raise ValueError(f"Unknown BC type: {bc_type}")
