
import numpy as np
from scipy.interpolate import RBFInterpolator

def make_smooth_interpolant(
    x_centers,
    y_centers,
    grid,
    smoothing=0.2,
    neighbors=20,
    kernel="thin_plate_spline",
):
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    grid = np.asarray(grid)

    mask = ~np.isnan(grid)
    if not np.any(mask):
        raise ValueError("Grid has no finite values to build an interpolant.")

    Xg, Yg = np.meshgrid(x_centers, y_centers, indexing="ij")
    pts = np.column_stack([Xg[mask], Yg[mask]])
    vals = grid[mask]

    rbf = RBFInterpolator(
        pts,
        vals,
        smoothing=smoothing,
        neighbors=neighbors,
        kernel=kernel,
    )

    def f(x, y):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        Xb, Yb = np.broadcast_arrays(x_arr, y_arr)
        pts_eval = np.column_stack([Xb.ravel(), Yb.ravel()])
        return rbf(pts_eval).reshape(Xb.shape)

    return f
