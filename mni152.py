"""Orthogonal slices of one NIfTI volume, in 2D and in place in 3D.

Four cells: axial, coronal, sagittal, and the 3D planes with their sliders.
Scroll or drag a panel to move that slice. The sliders do the same.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import patheffects
from matplotlib.image import _resample
from matplotlib.patches import Patch
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parent

# Pick one volume.
# https://niivue.github.io/niivue-demo-images/mni152.nii.gz
# PATH = ROOT / "mni152.nii"
# PATH = ROOT / "icbm_avg_152_t1_tal_lin.nii"
# PATH = ROOT / "minimal.nii"
# PATH = ROOT / "avg152T1_RL_nifti.nii"
# PATH = ROOT / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
PATH = ROOT / "mni_icbm152_t1_tal_nlin_asym_09b_hires.nii"

PLANES = (
    ("Sagittal", 0, "#c44747"),
    ("Coronal", 1, "#2f9a62"),
    ("Axial", 2, "#3a6fbe"),
)
PANEL_ORDER = (2, 1, 0)
# Image left on these panels is subject left.
LETTER_AXES = (1, 2)
SETTLE_MS = 80


class _ResampleStyle:
    """Flags for matplotlib's image resampler."""

    def get_interpolation(self):
        return "auto"

    def get_resample(self):
        return False

    def get_filternorm(self):
        return True

    def get_filterrad(self):
        return 4.0


_SAMPLER = _ResampleStyle()


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


def _plane_rgba(data, axis, index, lo, hi, fade, solid):
    """Slice as RGBA. Row is the second in-plane axis, column the first."""
    image = _slice(data, axis, index)
    gray = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    alpha = np.clip((image - fade) / (solid - fade), 0.0, 1.0)
    rgba = np.empty((image.shape[1], image.shape[0], 4), dtype=np.float32)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = gray.T
    rgba[..., 3] = alpha.T
    return rgba


def _spans(length, cut):
    """Voxel ranges on either side of an intersecting plane."""
    cut = int(cut)
    if cut <= 0 or cut >= length:
        return [(0, length)]
    return [(0, cut), (cut, length)]


def _rect_corners(affine, axis, index, u_axis, v_axis, ua, ub, va, vb):
    # (ua, va), (ub, va), (ua, vb), (ub, vb). ua/ub are voxel edges.
    u = (ua, ub, ua, ub)
    v = (va, va, vb, vb)
    ijk = [np.zeros(4), np.zeros(4), np.zeros(4)]
    ijk[axis][:] = index
    ijk[u_axis][:] = u
    ijk[v_axis][:] = v
    x, y, z = ijk_to_world(affine, *ijk)
    return np.column_stack((x, y, z))


def _affine(src, dst):
    rows = []
    target = []
    for (x, y), (out_x, out_y) in zip(src, dst):
        rows.append((x, y, 1.0, 0.0, 0.0, 0.0))
        rows.append((0.0, 0.0, 0.0, x, y, 1.0))
        target.append(out_x)
        target.append(out_y)
    a, c, e, b, d, f = np.linalg.solve(
        np.asarray(rows, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    return Affine2D.from_values(a, b, c, d, e, f)


class _PlanePiece(Patch):
    """One quadrant of a slice, drawn as a single image.

    Quadrants meet on the other two planes. Painter's sort then lets the
    three slices cut through each other instead of one card hiding another.
    """

    def __init__(self):
        super().__init__()
        self.set_linewidth(0)
        self._image = None
        self._corners = None
        self._pts = None
        self._pixmap = None
        self._origin = None
        self._key = None
        self.set_visible(False)

    def get_path(self):
        # add_patch asks for a 2D path. The slice is drawn in draw().
        return MPath([(0.0, 0.0), (0.0, 0.0)])

    def set_piece(self, image, corners):
        self._image = np.ascontiguousarray(image)
        self._corners = corners
        self._pixmap = None
        self._key = None
        self.set_visible(True)
        self.stale = True

    def clear_piece(self):
        self._image = None
        self._corners = None
        self._pixmap = None
        self._key = None
        self.set_visible(False)
        self.stale = True

    def do_3d_projection(self):
        corners = self._corners
        if corners is None:
            self._pts = None
            return 0.0
        txs, tys, tzs = proj3d.proj_transform(
            corners[:, 0], corners[:, 1], corners[:, 2], self.axes.M,
        )
        self._pts = np.column_stack((txs, tys))
        return float(np.mean(tzs))

    def draw(self, renderer):
        if not self.get_visible() or self._pts is None or self._image is None:
            self.stale = False
            return
        disp = self.axes.transData.transform(self._pts)
        key = tuple(np.round(disp.ravel()))
        gc = renderer.new_gc()
        gc.set_clip_rectangle(self.axes.bbox)
        if self._pixmap is not None and key == self._key:
            renderer.draw_image(gc, self._origin[0], self._origin[1], self._pixmap)
            gc.restore()
            self.stale = False
            return

        axbb = self.axes.bbox
        bb0 = np.maximum(np.floor(disp.min(axis=0) - 1.0), (axbb.x0, axbb.y0))
        bb1 = np.minimum(np.ceil(disp.max(axis=0) + 1.0), (axbb.x1, axbb.y1))
        out_w = int(bb1[0] - bb0[0])
        out_h = int(bb1[1] - bb0[1])
        if out_w < 2 or out_h < 2:
            gc.restore()
            self.stale = False
            return

        height, width = self._image.shape[:2]
        src = ((-0.5, -0.5), (width - 0.5, -0.5), (-0.5, height - 0.5))
        out = disp[:3] - bb0
        sampled = _resample(_SAMPLER, self._image, (out_h, out_w), _affine(src, out))
        pixmap = np.ascontiguousarray(np.clip(np.rint(sampled * 255.0), 0, 255).astype(np.uint8))
        renderer.draw_image(gc, float(bb0[0]), float(bb0[1]), pixmap)
        gc.restore()
        self._pixmap = pixmap
        self._origin = (float(bb0[0]), float(bb0[1]))
        self._key = key
        self.stale = False


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


def _axis_mm(affine, axis, index):
    column = affine[:3, axis]
    length = np.linalg.norm(column)
    if length == 0 or abs(column[axis]) < 0.9 * length:
        return None
    return float(affine[axis, 3] + column[axis] * index)


def _crosshair_geometry(shape, axis):
    u_axis, v_axis = [a for a in range(3) if a != axis]
    return u_axis, v_axis, shape[u_axis], shape[v_axis]


def _position_text(affine, axis, index):
    mm = _axis_mm(affine, axis, index)
    if mm is None:
        return str(int(index))
    return f"{'xyz'[axis]} = {mm:.0f} mm"


def draw(data, affine):
    lo, hi, fade, solid = display_levels(data)
    indices = index_at_world_zero(affine, data.shape)
    box = tissue_box(data, solid)

    fig = plt.figure(figsize=(11.6, 11.2))
    fig_w, fig_h = fig.get_size_inches()
    # Equal cells in a cluster, with room outside the group. Left-column
    # views sit on the gutter; right-column views do the same.
    margin_x = 1.15
    margin_top = 0.56
    margin_bottom = 0.30
    gap = 0.42
    cell_w = (fig_w - 2 * margin_x - gap) / 2
    cell_h = (fig_h - margin_top - margin_bottom - gap) / 2

    def cell_box(col, row):
        x = margin_x + col * (cell_w + gap)
        y = margin_bottom + (1 - row) * (cell_h + gap)
        return x, y

    panels = []
    title_in = 0.36
    for axis, (col, row) in zip(PANEL_ORDER, ((0, 0), (1, 0), (0, 1))):
        cx, cy = cell_box(col, row)
        _u_axis, _v_axis, n_u, n_v = _crosshair_geometry(data.shape, axis)
        aspect = n_u / max(n_v, 1)
        avail_w = cell_w
        avail_h = cell_h - title_in
        if avail_w / max(avail_h, 0.1) > aspect:
            height = avail_h
            width = height * aspect
        else:
            width = avail_w
            height = width / aspect
        # Face the neighboring column instead of the window edge.
        x = cx + cell_w - width if col == 0 else cx
        ax = fig.add_axes([
            x / fig_w,
            (cy + cell_h - title_in - height) / fig_h,
            width / fig_w,
            height / fig_h,
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
        name, _axis, color = PLANES[axis]
        ax.set_title(name, fontsize=12, color=color, pad=4)
        ax.axis("off")
        ax.set_navigate(False)
        mm = ax.text(
            0.5, 0.975, _position_text(affine, axis, indices[axis]),
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, color=color,
        )
        letter = None
        if axis in LETTER_AXES:
            letter = ax.text(
                0.035, 0.50, "L",
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=13, color="white", fontweight="medium",
                path_effects=[patheffects.withStroke(linewidth=2.4, foreground=(0, 0, 0, 0.7))],
                zorder=4,
            )
        panels.append({
            "axis": axis,
            "artist": artist,
            "v_line": v_line,
            "h_line": h_line,
            "u_axis": u_axis,
            "v_axis": v_axis,
            "n_v": n_v,
            "axes": ax,
            "mm": mm,
            "letter": letter,
        })

    cx, cy = cell_box(1, 1)
    slider_h = 0.20
    slider_gap = 0.08
    slider_block = 3 * slider_h + 2 * slider_gap
    plot_top = cy + cell_h - title_in
    plot_floor = cy + 0.02 + slider_block + 0.46
    # Inset so the z label and the slider names stay clear of the sagittal cell.
    inset = 0.34
    side = min(cell_w - inset - 0.62, plot_top - plot_floor)
    ax3 = fig.add_axes([
        (cx + inset) / fig_w,
        (plot_top - side) / fig_h,
        side / fig_w,
        side / fig_h,
    ], projection="3d")
    pieces = [_PlanePiece() for _ in range(12)]
    for piece in pieces:
        ax3.add_patch(piece)

    borders = []
    for name_axis in PLANES:
        _name, axis, color = name_axis
        outline = _frame(affine, box, axis, indices[axis])
        line, = ax3.plot(
            outline[0], outline[1], outline[2],
            color=color, linewidth=1.6, solid_capstyle="round", zorder=1000,
        )
        borders.append(line)

    low, high = volume_bounds(affine, data.shape)
    span = np.maximum(high - low, 1.0)
    pad = span * 0.03
    ax3.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax3.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax3.set_zlim(low[2] - pad[2], high[2] + pad[2])
    ax3.set_box_aspect(tuple(span), zoom=1.12)
    ax3.set_autoscale_on(False)
    ax3.set_proj_type("ortho")
    ax3.view_init(elev=28, azim=40)
    ax3.mouse_init(rotate_btn=[])
    ax3.set_xlabel("x (mm)", fontsize=8, labelpad=2)
    ax3.set_ylabel("y (mm)", fontsize=8, labelpad=2)
    ax3.set_zlabel("z (mm)", fontsize=8, labelpad=2)
    ax3.tick_params(labelsize=7, pad=0)
    ax3.grid(False)
    for axis3d in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis3d.pane.fill = False
        axis3d.pane.set_edgecolor((0, 0, 0, 0))

    sliders = []
    for slot, (name, axis, color) in enumerate(PLANES):
        ax_s = fig.add_axes([
            (cx + inset) / fig_w,
            (cy + 0.04 + (2 - slot) * (slider_h + slider_gap)) / fig_h,
            side / fig_w,
            slider_h / fig_h,
        ])
        slider = Slider(
            ax_s,
            name,
            0,
            max(data.shape[axis] - 1, 1),
            valinit=indices[axis],
            valstep=1,
            color=color,
            handle_style={"size": 8},
        )
        slider.drawon = False
        slider.label.set_color(color)
        slider.label.set_fontsize(9)
        slider.valtext.set_color(color)
        slider.valtext.set_fontsize(9)
        sliders.append(slider)

    shown = {"ready": False}

    def refresh_2d():
        current = [int(slider.val) for slider in sliders]
        for panel in panels:
            axis = panel["axis"]
            panel["artist"].set_data(np.rot90(_slice(data, axis, current[axis])))
            x = current[panel["u_axis"]]
            y = panel["n_v"] - 1 - current[panel["v_axis"]]
            panel["v_line"].set_xdata([x, x])
            panel["h_line"].set_ydata([y, y])
            panel["mm"].set_text(_position_text(affine, axis, current[axis]))
        for slider, (_name, axis, _color) in zip(sliders, PLANES):
            slider.valtext.set_text(_position_text(affine, axis, current[axis]))
        # The slice and the sliders repaint in place. The 3D planes stay as
        # they were until the drag settles, so this does not rebuild them.
        if not shown["ready"]:
            fig.canvas.draw_idle()
            return
        renderer = fig.canvas.get_renderer()
        for panel in panels:
            panel["axes"].draw(renderer)
        for slider in sliders:
            slider.ax.draw(renderer)
        fig.canvas.blit(fig.bbox)

    def refresh_3d():
        current = [int(slider.val) for slider in sliders]
        for axis, (_name, _axis, _color) in enumerate(PLANES):
            u_axis, v_axis, n_u, n_v = _crosshair_geometry(data.shape, axis)
            rgba = _plane_rgba(data, axis, current[axis], lo, hi, fade, solid)
            rects = [
                (ua, ub, va, vb)
                for ua, ub in _spans(n_u, current[u_axis])
                for va, vb in _spans(n_v, current[v_axis])
            ]
            borders[axis].set_data_3d(*_frame(affine, box, axis, current[axis]))
            for slot in range(4):
                piece = pieces[axis * 4 + slot]
                if slot >= len(rects):
                    piece.clear_piece()
                    continue
                ua, ub, va, vb = rects[slot]
                corners = _rect_corners(
                    affine, axis, current[axis], u_axis, v_axis,
                    ua - 0.5, ub - 0.5, va - 0.5, vb - 0.5,
                )
                piece.set_piece(rgba[va:vb, ua:ub], corners)
        fig.canvas.draw_idle()

    pending = {"dirty": False}
    timer = fig.canvas.new_timer(interval=SETTLE_MS)
    timer.single_shot = True

    def flush_3d():
        if not pending["dirty"]:
            return
        pending["dirty"] = False
        timer.stop()
        refresh_3d()

    def schedule_3d():
        pending["dirty"] = True
        timer.stop()
        timer.start()

    timer.add_callback(flush_3d)

    def on_slide(_value):
        refresh_2d()
        schedule_3d()

    for slider in sliders:
        slider.on_changed(on_slide)

    def set_plane(axis, index):
        index = int(np.clip(round(index), 0, data.shape[axis] - 1))
        if int(sliders[axis].val) == index:
            return
        sliders[axis].set_val(index)

    drag = {"axis": None, "y": 0.0, "index": 0, "height": 1.0}

    def panel_at(inaxes):
        for panel in panels:
            if panel["axes"] is inaxes:
                return panel
        return None

    def on_scroll(event):
        panel = panel_at(event.inaxes)
        if panel is None or not event.step:
            return
        step = 1 if event.step > 0 else -1
        set_plane(panel["axis"], int(sliders[panel["axis"]].val) + step)

    def on_press(event):
        if event.button != 1 or event.y is None:
            return
        panel = panel_at(event.inaxes)
        if panel is None:
            return
        drag["axis"] = panel["axis"]
        drag["y"] = event.y
        drag["index"] = int(sliders[panel["axis"]].val)
        drag["height"] = max(panel["axes"].bbox.height, 1.0)

    def on_motion(event):
        axis = drag["axis"]
        if axis is None or event.y is None:
            return
        # A drag the height of the panel walks the whole stack.
        span = max(data.shape[axis] - 1, 1)
        delta = (event.y - drag["y"]) / drag["height"] * span
        set_plane(axis, drag["index"] + delta)

    def on_release(_event):
        if drag["axis"] is None and not pending["dirty"]:
            return
        drag["axis"] = None
        flush_3d()

    def on_draw(_event):
        shown["ready"] = True

    fig.canvas.mpl_connect("draw_event", on_draw)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig._slice_timer = timer
    # Sliders are only weakly referenced by the canvas. Keep them with the figure.
    fig._slice_sliders = sliders

    for slider, (_name, axis, _color) in zip(sliders, PLANES):
        slider.valtext.set_text(_position_text(affine, axis, indices[axis]))
    refresh_3d()
    pending["dirty"] = False
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
