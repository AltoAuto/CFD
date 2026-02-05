from pathlib import Path
import numpy as np

def _segment_nodes(length, count, expansion_ratio):
    """Return node coordinates for a geometric segment with given ratio."""
    if count <= 0:
        raise ValueError("segment cell count must be positive")
    if length <= 0.0:
        raise ValueError("segment length must be positive")
    if expansion_ratio <= 0.0:
        raise ValueError("segment expansion ratio must be positive")

    if count == 1:
        return np.array([0.0, length], dtype=float)

    if abs(expansion_ratio - 1.0) < 1.0e-12:
        return np.linspace(0.0, length, count + 1)

    ratio = expansion_ratio
    d0 = length * (1.0 - ratio) / (1.0 - ratio ** count)
    widths = d0 * ratio ** np.arange(count, dtype=float)
    nodes = np.concatenate(([0.0], np.cumsum(widths)))
    nodes[-1] = length
    return nodes


def _segment_counts(total_count, segments):
    """Split a total cell count across fractional segments."""
    fractions = np.array([seg[0] for seg in segments], dtype=float)
    cumulative = np.cumsum(fractions)
    boundaries = np.round(cumulative * total_count).astype(int)
    counts = np.diff(np.concatenate(([0], boundaries)))
    if counts.sum() != total_count:
        counts[-1] += total_count - counts.sum()
    if np.any(counts <= 0):
        raise ValueError("segment has zero cells; increase count or adjust fractions")
    return counts


def _geometric_nodes(length, count, ratio):
    """Generate 1D node positions with optional geometric expansion."""
    if count <= 0:
        raise ValueError("count must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    if isinstance(ratio, (list, tuple)) and ratio and isinstance(ratio[0], (list, tuple)):
        segments = [(float(frac), float(exp)) for frac, exp in ratio]
        total = sum(frac for frac, _ in segments)
        if abs(total - 1.0) > 1.0e-12:
            raise ValueError("segment fractions must sum to 1.0")
        for frac, exp in segments:
            if frac <= 0.0:
                raise ValueError("segment fraction must be positive")
            if exp <= 0.0:
                raise ValueError("segment expansion ratio must be positive")

        counts = _segment_counts(count, segments)
        nodes = [0.0]
        offset = 0.0
        for (frac, exp), seg_count in zip(segments, counts):
            seg_len = length * frac
            seg_nodes = _segment_nodes(seg_len, seg_count, exp)
            nodes.extend((seg_nodes[1:] + offset).tolist())
            offset += seg_len
        nodes[-1] = length
        return np.array(nodes, dtype=float)

    if ratio <= 0.0:
        raise ValueError("ratio must be positive")

    if abs(ratio - 1.0) < 1.0e-12:
        return np.linspace(0.0, length, count + 1)

    d0 = length * (1.0 - ratio) / (1.0 - ratio ** count)
    widths = d0 * ratio ** np.arange(count, dtype=float)
    nodes = np.concatenate(([0.0], np.cumsum(widths)))
    nodes[-1] = length
    return nodes


def generate_rect_nodes(length_i, length_j, count_i, count_j, ratio_i=1.0, ratio_j=1.0):
    """Generate rectangular grid nodes in physical space."""
    nodes_i = _geometric_nodes(length_i, count_i, ratio_i)
    nodes_j = _geometric_nodes(length_j, count_j, ratio_j)
    ii, jj = np.meshgrid(nodes_i, nodes_j, indexing="ij")
    return np.stack((ii, jj), axis=-1)


def generate_annulus_nodes(
    radius_inner,
    radius_outer,
    count_r,
    count_t,
    theta_min=0.0,
    theta_max=np.pi,
    ratio_r=1.0,
    ratio_t=1.0,
):
    """Generate concentric annulus nodes in (x,y) for given radii and angles."""
    if radius_inner <= 0.0 or radius_outer <= 0.0:
        raise ValueError("radii must be positive")
    if radius_outer <= radius_inner:
        raise ValueError("radius_outer must be greater than radius_inner")
    if count_r <= 0 or count_t <= 0:
        raise ValueError("cell counts must be positive")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")

    r_nodes = _geometric_nodes(radius_outer - radius_inner, count_r, ratio_r) + radius_inner
    t_nodes = _geometric_nodes(theta_max - theta_min, count_t, ratio_t) + theta_min

    rr, tt = np.meshgrid(r_nodes, t_nodes, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    return np.stack((x, y), axis=-1)


def ensure_single_mesh_file(mesh_path):
    """Ensure the mesh directory only contains the specified mesh file."""
    mesh_path = Path(mesh_path)
    mesh_dir = mesh_path.parent
    if not mesh_dir.exists():
        return
    candidates = sorted(mesh_dir.glob("*.npz"))
    others = [p for p in candidates if p.resolve() != mesh_path.resolve()]
    if others:
        names = ", ".join(p.name for p in others)
        raise ValueError(
            f"Multiple mesh files found in {mesh_dir}: {names}. Remove extras."
        )


def generate_ramp_nodes(
    length_i,
    height_in,
    height_out,
    count_i,
    count_j,
    ramp_start,
    ramp_end,
    ratio_i=1.0,
    ratio_j=1.0,
):
    """Generate ramp channel nodes with linear height change in i."""
    if length_i <= 0.0:
        raise ValueError("length_i must be positive")
    if height_in <= 0.0 or height_out <= 0.0:
        raise ValueError("heights must be positive")
    if ramp_start < 0.0 or ramp_end < 0.0:
        raise ValueError("ramp positions must be non-negative")
    if ramp_end <= ramp_start:
        raise ValueError("ramp_end must be greater than ramp_start")
    if ramp_end > length_i:
        raise ValueError("ramp_end must be within the domain length")

    nodes_i = _geometric_nodes(length_i, count_i, ratio_i)
    eta = _geometric_nodes(1.0, count_j, ratio_j)

    height = np.empty_like(nodes_i)
    before = nodes_i <= ramp_start
    after = nodes_i >= ramp_end
    height[before] = height_in
    height[after] = height_out
    ramp_mask = ~(before | after)
    if np.any(ramp_mask):
        t = (nodes_i[ramp_mask] - ramp_start) / (ramp_end - ramp_start)
        height[ramp_mask] = height_in + t * (height_out - height_in)

    node = np.zeros((count_i + 1, count_j + 1, 2), dtype=float)
    node[:, :, 0] = nodes_i[:, None]
    node[:, :, 1] = eta[None, :] * height[:, None]
    return node


def make_mesh(mesh_cfg):
    """Generate and save a mesh using the configured generator."""
    path = Path(mesh_cfg["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_single_mesh_file(path)
    if path.exists():
        path.unlink()
    return mesh_cfg["generator"](out_path=str(path), **mesh_cfg["params"])


def precompute_from_nodes(node, extra_fields=None):
    """Precompute cell and face geometry from curvilinear nodes."""
    if node.ndim != 3 or node.shape[2] != 2:
        raise ValueError("node must have shape (count_i+1, count_j+1, 2)")

    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    x = node[..., 0]
    y = node[..., 1]

    x00 = x[:-1, :-1]
    y00 = y[:-1, :-1]
    x10 = x[1:, :-1]
    y10 = y[1:, :-1]
    x11 = x[1:, 1:]
    y11 = y[1:, 1:]
    x01 = x[:-1, 1:]
    y01 = y[:-1, 1:]

    cross0 = x00 * y10 - x10 * y00
    cross1 = x10 * y11 - x11 * y10
    cross2 = x11 * y01 - x01 * y11
    cross3 = x01 * y00 - x00 * y01

    area_signed = 0.5 * (cross0 + cross1 + cross2 + cross3)
    cell_volume = np.abs(area_signed)

    cx_num = (
        (x00 + x10) * cross0
        + (x10 + x11) * cross1
        + (x11 + x01) * cross2
        + (x01 + x00) * cross3
    )
    cy_num = (
        (y00 + y10) * cross0
        + (y10 + y11) * cross1
        + (y11 + y01) * cross2
        + (y01 + y00) * cross3
    )
    cell_center_x = cx_num / (6.0 * area_signed)
    cell_center_y = cy_num / (6.0 * area_signed)
    cell_center = np.stack((cell_center_x, cell_center_y), axis=-1)

    edge_i_lower = node[1:, :-1] - node[:-1, :-1]
    edge_i_upper = node[1:, 1:] - node[:-1, 1:]
    edge_j_left = node[:-1, 1:] - node[:-1, :-1]
    edge_j_right = node[1:, 1:] - node[1:, :-1]

    face_center = np.zeros((count_i, count_j, 2, 2), dtype=float)
    face_center[:, :, 0, :] = 0.5 * (node[:-1, :-1] + node[:-1, 1:])
    face_center[:, :, 1, :] = 0.5 * (node[:-1, :-1] + node[1:, :-1])

    face_size = np.zeros((count_i, count_j, 2), dtype=float)
    face_size[:, :, 0] = np.hypot(edge_j_left[..., 0], edge_j_left[..., 1])
    face_size[:, :, 1] = np.hypot(edge_i_lower[..., 0], edge_i_lower[..., 1])

    cell_spacing_i = 0.5 * (
        np.hypot(edge_i_lower[..., 0], edge_i_lower[..., 1])
        + np.hypot(edge_i_upper[..., 0], edge_i_upper[..., 1])
    )
    cell_spacing_j = 0.5 * (
        np.hypot(edge_j_left[..., 0], edge_j_left[..., 1])
        + np.hypot(edge_j_right[..., 0], edge_j_right[..., 1])
    )

    face_normal = np.zeros((count_i, count_j, 2, 2), dtype=float)

    t_i = edge_j_left
    n_i = np.stack((t_i[..., 1], -t_i[..., 0]), axis=-1)
    s_i = 0.5 * (edge_i_lower + edge_i_upper)
    dot_i = np.sum(n_i * s_i, axis=-1)
    n_i *= np.where(dot_i < 0.0, -1.0, 1.0)[..., None]
    face_normal[:, :, 0, :] = n_i

    t_j = edge_i_lower
    n_j = np.stack((-t_j[..., 1], t_j[..., 0]), axis=-1)
    s_j = 0.5 * (edge_j_left + edge_j_right)
    dot_j = np.sum(n_j * s_j, axis=-1)
    n_j *= np.where(dot_j < 0.0, -1.0, 1.0)[..., None]
    face_normal[:, :, 1, :] = n_j

    def _linear_face_weights(left, right, face):
        d = right - left
        denom = d[..., 0] ** 2 + d[..., 1] ** 2
        denom_safe = np.where(denom > 1.0e-14, denom, 1.0)
        t = ((face - left) * d).sum(axis=-1) / denom_safe
        t = np.clip(t, 0.0, 1.0)
        w_right = t
        w_left = 1.0 - t
        return w_left, w_right

    w_left_i = np.full((count_i + 1, count_j), 0.5, dtype=float)
    w_right_i = np.full((count_i + 1, count_j), 0.5, dtype=float)
    left = cell_center[:-1, :, :]
    right = cell_center[1:, :, :]
    face_i = face_center[1:, :, 0, :]
    wl, wr = _linear_face_weights(left, right, face_i)
    w_left_i[1:count_i, :] = wl
    w_right_i[1:count_i, :] = wr

    w_south_j = np.full((count_i, count_j + 1), 0.5, dtype=float)
    w_north_j = np.full((count_i, count_j + 1), 0.5, dtype=float)
    south = cell_center[:, :-1, :]
    north = cell_center[:, 1:, :]
    face_j = face_center[:, 1:, 1, :]
    ws, wn = _linear_face_weights(south, north, face_j)
    w_south_j[:, 1:count_j] = ws
    w_north_j[:, 1:count_j] = wn

    face_center_i = np.zeros((count_i + 1, count_j, 2), dtype=float)
    face_center_j = np.zeros((count_i, count_j + 1, 2), dtype=float)
    face_center_i[:count_i, :, :] = face_center[:, :, 0, :]
    face_center_j[:, :count_j, :] = face_center[:, :, 1, :]
    n0 = node[count_i, 0:count_j, :]
    n1 = node[count_i, 1 : count_j + 1, :]
    face_center_i[count_i, :, :] = 0.5 * (n0 + n1)
    n0 = node[0:count_i, count_j, :]
    n1 = node[1 : count_i + 1, count_j, :]
    face_center_j[:, count_j, :] = 0.5 * (n0 + n1)

    face_normal_i = np.zeros((count_i + 1, count_j, 2), dtype=float)
    face_normal_j = np.zeros((count_i, count_j + 1, 2), dtype=float)
    face_normal_i[:count_i, :, :] = face_normal[:, :, 0, :]
    face_normal_j[:, :count_j, :] = face_normal[:, :, 1, :]

    n0 = node[count_i, 0:count_j, :]
    n1 = node[count_i, 1 : count_j + 1, :]
    t = n1 - n0
    n = np.stack((t[..., 1], -t[..., 0]), axis=-1)
    fc = 0.5 * (n0 + n1)
    cc = cell_center[count_i - 1, :, :]
    dot = np.sum(n * (fc - cc), axis=-1)
    n *= np.where(dot < 0.0, -1.0, 1.0)[..., None]
    face_normal_i[count_i, :, :] = n

    n0 = node[0:count_i, count_j, :]
    n1 = node[1 : count_i + 1, count_j, :]
    t = n1 - n0
    n = np.stack((-t[..., 1], t[..., 0]), axis=-1)
    fc = 0.5 * (n0 + n1)
    cc = cell_center[:, count_j - 1, :]
    dot = np.sum(n * (fc - cc), axis=-1)
    n *= np.where(dot < 0.0, -1.0, 1.0)[..., None]
    face_normal_j[:, count_j, :] = n

    s_i = np.hypot(face_normal_i[..., 0], face_normal_i[..., 1])
    s_j = np.hypot(face_normal_j[..., 0], face_normal_j[..., 1])
    n_i = face_normal_i / np.where(s_i > 1.0e-14, s_i, 1.0)[..., None]
    n_j = face_normal_j / np.where(s_j > 1.0e-14, s_j, 1.0)[..., None]

    diff_d_i = np.zeros((count_i + 1, count_j), dtype=float)
    diff_d_j = np.zeros((count_i, count_j + 1), dtype=float)
    delta_i = cell_center[1:, :, :] - cell_center[:-1, :, :]
    diff_d_i[1:count_i, :] = np.abs(
        np.sum(delta_i * n_i[1:count_i, :, :], axis=-1)
    )
    delta_j = cell_center[:, 1:, :] - cell_center[:, :-1, :]
    diff_d_j[:, 1:count_j] = np.abs(
        np.sum(delta_j * n_j[:, 1:count_j, :], axis=-1)
    )
    s_w = np.abs(
        np.sum((cell_center[0, :, :] - face_center_i[0, :, :]) * n_i[0, :, :], axis=-1)
    )
    diff_d_i[0, :] = 2.0 * s_w
    s_e = np.abs(
        np.sum(
            (cell_center[count_i - 1, :, :] - face_center_i[count_i, :, :])
            * n_i[count_i, :, :],
            axis=-1,
        )
    )
    diff_d_i[count_i, :] = 2.0 * s_e
    s_s = np.abs(
        np.sum((cell_center[:, 0, :] - face_center_j[:, 0, :]) * n_j[:, 0, :], axis=-1)
    )
    diff_d_j[:, 0] = 2.0 * s_s
    s_n = np.abs(
        np.sum(
            (cell_center[:, count_j - 1, :] - face_center_j[:, count_j, :])
            * n_j[:, count_j, :],
            axis=-1,
        )
    )
    diff_d_j[:, count_j] = 2.0 * s_n

    diff_d_i = np.where(diff_d_i > 1.0e-14, diff_d_i, 1.0e-14)
    diff_d_j = np.where(diff_d_j > 1.0e-14, diff_d_j, 1.0e-14)

    mesh = {
        "node": node,
        "cell_center": cell_center,
        "face_center": face_center,
        "cell_volume": cell_volume,
        "face_size": face_size,
        "face_normal": face_normal,
        "cell_spacing_i": cell_spacing_i,
        "cell_spacing_j": cell_spacing_j,
        "w_left_i": w_left_i,
        "w_right_i": w_right_i,
        "w_south_j": w_south_j,
        "w_north_j": w_north_j,
        "diff_d_i": diff_d_i,
        "diff_d_j": diff_d_j,
        "face_normal_i": face_normal_i,
        "face_normal_j": face_normal_j,
        "count_i": np.array(count_i, dtype=int),
        "count_j": np.array(count_j, dtype=int),
    }

    if extra_fields:
        mesh.update(extra_fields)

    return mesh


def generate_rect_mesh(
    length_i,
    length_j,
    count_i,
    count_j,
    ratio_i=1.0,
    ratio_j=1.0,
    out_path=None,
    extra_fields=None,
):
    """Generate a rectangular mesh and optionally save to disk."""
    node = generate_rect_nodes(length_i, length_j, count_i, count_j, ratio_i, ratio_j)
    mesh = precompute_from_nodes(node, extra_fields=extra_fields)

    if out_path:
        np.savez(out_path, **mesh)

    return mesh


def generate_ramp_mesh(
    length_i,
    height_in,
    height_out,
    count_i,
    count_j,
    ramp_start,
    ramp_end,
    ratio_i=1.0,
    ratio_j=1.0,
    out_path=None,
    extra_fields=None,
):
    """Generate a ramp mesh and optionally save to disk."""
    node = generate_ramp_nodes(
        length_i,
        height_in,
        height_out,
        count_i,
        count_j,
        ramp_start,
        ramp_end,
        ratio_i,
        ratio_j,
    )
    mesh = precompute_from_nodes(node, extra_fields=extra_fields)

    if out_path:
        np.savez(out_path, **mesh)

    return mesh


def generate_annulus_mesh(
    radius_inner,
    radius_outer,
    count_r,
    count_t,
    theta_min=0.0,
    theta_max=np.pi,
    ratio_r=1.0,
    ratio_t=1.0,
    out_path=None,
    extra_fields=None,
):
    """Generate an annulus mesh and optionally save to disk."""
    node = generate_annulus_nodes(
        radius_inner,
        radius_outer,
        count_r,
        count_t,
        theta_min=theta_min,
        theta_max=theta_max,
        ratio_r=ratio_r,
        ratio_t=ratio_t,
    )
    mesh = precompute_from_nodes(node, extra_fields=extra_fields)

    if out_path:
        np.savez(out_path, **mesh)

    return mesh
