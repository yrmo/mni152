"""Orthogonal slices of one NIfTI volume, in 2D and in place in 3D.

The 2D panels are the voxels. The 3D view is those same three slices, moved
with the sliders. Its textures are a coarse copy so rotation stays light.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import path as mpath
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage

ROOT = Path(__file__).resolve().parent

# Pick one volume.
# https://niivue.github.io/niivue-demo-images/mni152.nii.gz
# PATH = ROOT / "mni152.nii" / "mni152.nii"
# PATH = ROOT / "icbm_avg_152_t1_tal_lin.nii" / "icbm_avg_152_t1_tal_lin.nii"
# PATH = ROOT / "minimal.nii" / "minimal.nii"
# PATH = ROOT / "avg152T1_RL_nifti.nii" / "avg152T1_RL_nifti.nii"
# PATH = ROOT / "mni_icbm152_t1_tal_nlin_asym_09c.nii" / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
PATH = ROOT / "mni_icbm152_t1_tal_nlin_asym_09b_hires.nii" / "mni_icbm152_t1_tal_nlin_asym_09b_hires.nii"

# Quads along each side of a 3D slice. The side panels stay full resolution.
# This grid is only the rotatable texture, kept coarse so dragging stays light.
PLANE_QUADS = 32

PLANES = (
    ("Sagittal", 0, "#c44747"),
    ("Coronal", 1, "#2f9a62"),
    ("Axial", 2, "#3a6fbe"),
)
PANEL_ORDER = (2, 1, 0)
_PATH_CODES = np.array(
    [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.CLOSEPOLY,
    ],
    dtype=mpath.Path.code_type,
)


def load(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)
    return data, affine


def display_levels(data):
    """Gray window, and the air range that is drawn transparent in 3D."""
    sample = data[::4, ::4, ::4]
    brain = sample[sample > 0]
    if brain.size == 0:
        brain = sample.reshape(-1)
    lo, hi = np.percentile(brain, (1.0, 99.0))
    fade, solid = np.percentile(sample, (45.0, 52.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(sample))
        hi = float(np.max(sample))
    if hi <= lo:
        hi = lo + 1.0
    if not np.isfinite(fade) or not np.isfinite(solid) or solid <= fade:
        fade, solid = lo, lo + 1.0
    return float(lo), float(hi), float(fade), float(solid)


def index_at_world_zero(affine, shape):
    """Voxel nearest the world origin. For MNI templates that is AC."""
    try:
        ijk = np.linalg.solve(affine[:3, :3], -affine[:3, 3])
    except np.linalg.LinAlgError:
        ijk = [(n - 1) / 2 for n in shape]
    return [int(np.clip(np.rint(v), 0, n - 1)) for v, n in zip(ijk, shape)]


def ijk_to_world(affine, i, j, k):
    i = np.asarray(i, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    x = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + affine[0, 3]
    y = affine[1, 0] * i + affine[1, 1] * j + affine[1, 2] * k + affine[1, 3]
    z = affine[2, 0] * i + affine[2, 1] * j + affine[2, 2] * k + affine[2, 3]
    return x, y, z


def volume_bounds(affine, shape):
    ii, jj, kk = np.meshgrid(
        (0, shape[0] - 1),
        (0, shape[1] - 1),
        (0, shape[2] - 1),
        indexing="ij",
    )
    x, y, z = ijk_to_world(affine, ii, jj, kk)
    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def _slice(data, axis, index):
    if axis == 0:
        return data[index, :, :]
    if axis == 1:
        return data[:, index, :]
    return data[:, :, index]


def _plane(data, affine, axis, index, lo, hi, fade, solid):
    image = _slice(data, axis, index)
    u_axis, v_axis = [a for a in range(3) if a != axis]
    u = np.linspace(0, image.shape[0] - 1, PLANE_QUADS + 1)
    v = np.linspace(0, image.shape[1] - 1, PLANE_QUADS + 1)
    cu = (u[:-1] + u[1:]) * 0.5
    cv = (v[:-1] + v[1:]) * 0.5
    uu, vv = np.meshgrid(cu, cv, indexing="ij")
    sampled = ndimage.map_coordinates(
        image,
        np.stack((uu.ravel(), vv.ravel())),
        order=1,
        prefilter=False,
    ).reshape(PLANE_QUADS, PLANE_QUADS)
    gray = np.clip((sampled - lo) / (hi - lo), 0.0, 1.0)
    # Air is see-through. Scalp and brain stay opaque, still the voxel values.
    alpha = np.clip((sampled - fade) / (solid - fade), 0.0, 1.0)
    color = np.empty((gray.size, 4), dtype=np.float64)
    color[:, 0] = color[:, 1] = color[:, 2] = gray.ravel()
    color[:, 3] = alpha.ravel()

    ug, vg = np.meshgrid(u, v, indexing="ij")
    ijk = [None, None, None]
    ijk[u_axis] = ug
    ijk[v_axis] = vg
    ijk[axis] = np.full(ug.shape, float(index))
    x, y, z = ijk_to_world(affine, *ijk)
    xyz = np.stack((x, y, z), axis=-1)
    quads = np.stack(
        (xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, 1:], xyz[1:, :-1]),
        axis=2,
    ).reshape(-1, 4, 3)
    return quads, color


def tissue_box(data, solid):
    """Voxel bounds of the head, so the plane frames sit on the tissue."""
    step = 4
    sample = data[::step, ::step, ::step]
    hits = np.argwhere(sample >= solid)
    shape = np.asarray(data.shape)
    if hits.size == 0:
        return [(0, int(n - 1)) for n in shape]
    lo = hits.min(axis=0) * step
    hi = np.minimum(hits.max(axis=0) * step + step - 1, shape - 1)
    return [(int(a), int(b)) for a, b in zip(lo, hi)]


def _frame(affine, box, axis, index):
    u_axis, v_axis = [a for a in range(3) if a != axis]
    corners = (
        (box[u_axis][0], box[v_axis][0]),
        (box[u_axis][0], box[v_axis][1]),
        (box[u_axis][1], box[v_axis][1]),
        (box[u_axis][1], box[v_axis][0]),
        (box[u_axis][0], box[v_axis][0]),
    )
    ijk = np.zeros((5, 3), dtype=np.float64)
    ijk[:, axis] = index
    for n, (u, v) in enumerate(corners):
        ijk[n, u_axis] = u
        ijk[n, v_axis] = v
    x, y, z = ijk_to_world(affine, ijk[:, 0], ijk[:, 1], ijk[:, 2])
    return np.stack((x, y, z))


def _scene(data, affine, indices, lo, hi, fade, solid, box):
    quads = []
    colors = []
    frames = []
    for _name, axis, _color in PLANES:
        q, c = _plane(data, affine, axis, indices[axis], lo, hi, fade, solid)
        quads.append(q)
        colors.append(c)
        frames.append(_frame(affine, box, axis, indices[axis]))
    return np.concatenate(quads), np.concatenate(colors), frames


class _SlicePlanes(Poly3DCollection):
    """Slice quads with a vectorized depth sort and reused path objects."""

    def __init__(self, quads, facecolors):
        super().__init__(
            np.zeros((1, 4, 3)),
            facecolors=(0, 0, 0, 1),
            edgecolors=(0, 0, 0, 0),
            linewidths=0,
            antialiased=False,
            shade=False,
        )
        self._face_paths = None
        self.set_quads(quads, facecolors)

    def set_quads(self, quads, facecolors):
        flat = np.ascontiguousarray(quads, dtype=np.float64).reshape(-1, 3)
        vec = np.empty((4, flat.shape[0]), dtype=np.float64)
        vec[0] = flat[:, 0]
        vec[1] = flat[:, 1]
        vec[2] = flat[:, 2]
        vec[3] = 1.0
        self._vec = vec
        self._facecolor3d = np.ascontiguousarray(facecolors, dtype=np.float64)
        self._facecolors2d = self._facecolor3d
        self._edgecolors2d = self._facecolor3d
        self.stale = True

    def _sync_paths(self, pts):
        n = pts.shape[0]
        if self._face_paths is None or len(self._face_paths) != n:
            self._face_paths = [
                mpath.Path(np.zeros((5, 2)), _PATH_CODES.copy()) for _ in range(n)
            ]
        for path, quad in zip(self._face_paths, pts):
            vertex = path.vertices
            vertex[:4] = quad
            vertex[4] = quad[0]

    def do_3d_projection(self):
        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        n = tzs.shape[0] // 4
        pts = np.column_stack((txs, tys)).reshape(n, 4, 2)
        self._sync_paths(pts)
        order = np.argsort(-tzs.reshape(n, 4).mean(axis=1))
        self._paths = [self._face_paths[i] for i in order]
        self._facecolors2d = self._facecolor3d[order]
        self._edgecolors2d = self._facecolors2d
        return float(tzs.min())


def _axis_mm(affine, axis, index):
    column = affine[:3, axis]
    length = np.linalg.norm(column)
    if length == 0 or abs(column[axis]) < 0.9 * length:
        return None
    return float(affine[axis, 3] + column[axis] * index)


def _crosshair_geometry(shape, axis):
    u_axis, v_axis = [a for a in range(3) if a != axis]
    return u_axis, v_axis, shape[u_axis], shape[v_axis]


def draw(data, affine):
    lo, hi, fade, solid = display_levels(data)
    indices = index_at_world_zero(affine, data.shape)
    box = tissue_box(data, solid)
    quads, colors, outlines = _scene(data, affine, indices, lo, hi, fade, solid, box)

    fig = plt.figure(figsize=(12.4, 9.4))
    fig_w, fig_h = fig.get_size_inches()
    # Each panel is only as wide as its slice. A shared column grid left a
    # wide empty margin inside every axes, which pushed the brains apart.
    panel_height = 2.9
    panel_gap = 0.16
    panel_widths = []
    for axis in PANEL_ORDER:
        _u_axis, _v_axis, n_u, n_v = _crosshair_geometry(data.shape, axis)
        panel_widths.append(panel_height * n_u / max(n_v, 1))
    pack = sum(panel_widths) + panel_gap * (len(panel_widths) - 1)
    room = fig_w * 0.96
    if pack > room:
        scale = room / pack
        panel_height *= scale
        panel_gap *= scale
        panel_widths = [width * scale for width in panel_widths]
        pack = room
    panel_top = 0.90
    panel_bottom = panel_top - panel_height / fig_h
    x_in = (fig_w - pack) / 2.0

    panels = []
    for axis, width in zip(PANEL_ORDER, panel_widths):
        ax = fig.add_axes([
            x_in / fig_w,
            panel_bottom,
            width / fig_w,
            panel_height / fig_h,
        ])
        image = _slice(data, axis, indices[axis])
        n_u, n_v = image.shape
        artist = ax.imshow(
            np.rot90(image),
            cmap="gray",
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax.set_xlim(-0.5, n_u - 0.5)
        ax.set_ylim(n_v - 0.5, -0.5)
        u_axis, v_axis, _n_u, _n_v = _crosshair_geometry(data.shape, axis)
        v_line = ax.axvline(indices[u_axis], color=PLANES[u_axis][2], lw=0.9)
        h_line = ax.axhline(n_v - 1 - indices[v_axis], color=PLANES[v_axis][2], lw=0.9)
        ax.set_title(PLANES[axis][0], fontsize=11, color=PLANES[axis][2])
        ax.axis("off")
        panels.append((axis, artist, v_line, h_line, u_axis, v_axis, n_v))
        x_in += width + panel_gap

    plot_top = panel_bottom - 0.045
    ax3 = fig.add_axes([0.02, 0.15, 0.96, plot_top - 0.15], projection="3d")
    planes = _SlicePlanes(quads, colors)
    ax3.add_collection3d(planes)
    borders = []
    for outline, (_name, _axis, color) in zip(outlines, PLANES):
        line, = ax3.plot(
            outline[0],
            outline[1],
            outline[2],
            color=color,
            linewidth=1.6,
            solid_capstyle="round",
        )
        borders.append(line)

    low, high = volume_bounds(affine, data.shape)
    span = np.maximum(high - low, 1.0)
    pad = span * 0.03
    ax3.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax3.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax3.set_zlim(low[2] - pad[2], high[2] + pad[2])
    ax3.set_box_aspect(tuple(span), zoom=1.15)
    ax3.set_autoscale_on(False)
    ax3.set_proj_type("ortho")
    ax3.view_init(elev=18, azim=40)
    ax3.set_xlabel("x (mm)", fontsize=8)
    ax3.set_ylabel("y (mm)", fontsize=8)
    ax3.set_zlabel("z (mm)", fontsize=8)
    ax3.tick_params(labelsize=7, pad=0)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 1))
    ax3.set_title("Slice planes (drag to rotate)", fontsize=11, pad=0)

    sliders = []
    for slot, (name, axis, color) in enumerate(PLANES):
        ax_s = fig.add_axes([0.28, 0.098 - slot * 0.030, 0.44, 0.012])
        slider = Slider(
            ax_s,
            name,
            0,
            max(data.shape[axis] - 1, 1),
            valinit=indices[axis],
            valstep=1,
            color=color,
            handle_style={"size": 7},
        )
        slider.drawon = False
        slider.label.set_color(color)
        slider.label.set_fontsize(9)
        slider.valtext.set_color(color)
        slider.valtext.set_fontsize(9)
        sliders.append(slider)

    def show_value(slider, axis, index):
        mm = _axis_mm(affine, axis, index)
        letter = "xyz"[axis]
        if mm is None:
            slider.valtext.set_text(str(index))
        else:
            slider.valtext.set_text(f"{letter} = {mm:.0f} mm")

    for slider, (_name, axis, _color) in zip(sliders, PLANES):
        show_value(slider, axis, indices[axis])

    def refresh(new_indices):
        quads, colors, outlines = _scene(data, affine, new_indices, lo, hi, fade, solid, box)
        planes.set_quads(quads, colors)
        for line, outline in zip(borders, outlines):
            line.set_data_3d(outline[0], outline[1], outline[2])
        for axis, artist, v_line, h_line, u_axis, v_axis, n_v in panels:
            artist.set_data(np.rot90(_slice(data, axis, new_indices[axis])))
            x = new_indices[u_axis]
            y = n_v - 1 - new_indices[v_axis]
            v_line.set_xdata([x, x])
            h_line.set_ydata([y, y])
        for slider, (_name, axis, _color) in zip(sliders, PLANES):
            show_value(slider, axis, new_indices[axis])
        fig.canvas.draw_idle()

    def on_slide(_value):
        new_indices = [int(slider.val) for slider in sliders]
        refresh(new_indices)

    for slider in sliders:
        slider.on_changed(on_slide)

    fig.suptitle(PATH.name, fontsize=13)
    return fig


def main():
    if not PATH.exists():
        raise SystemExit(f"missing {PATH}")
    data, affine = load(PATH)
    draw(data, affine)
    plt.show()


if __name__ == "__main__":
    main()
