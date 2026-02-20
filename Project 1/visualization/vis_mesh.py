def view_mesh_pyvista(mesh_npz_path):
    import numpy as np
    import pyvista as pv

    data = np.load(mesh_npz_path, allow_pickle=True)

    node = data["node"]  # shape (Ni+1, Nj+1, 2)

    x = node[:, :, 0]
    y = node[:, :, 1]
    z = np.zeros_like(x)

    grid = pv.StructuredGrid(x, y, z)

    p = pv.Plotter()
    p.add_mesh(grid, show_edges=True)
    p.show()
