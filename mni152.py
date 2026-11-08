"""Orthogonal slices of one NIfTI volume, in 2D and in place in 3D.

Four cells: axial, coronal, sagittal, and the 3D planes with their sliders.
Scroll or drag a panel to move that slice. The sliders do the same.
A menu between the 3D view and the sliders opens another NIfTI file.
The control above the 3D view sets neurological or radiological display.
"""

from pathlib import Path
import tkinter

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import patheffects
from matplotlib.image import _resample
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parent

# Volume opened at startup. The menu under the 3D view lists every
# NIfTI file next to this script.
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
# Axial and coronal can show either convention. The L stays on subject left.
LETTER_AXES = (1, 2)
SETTLE_MS = 80


def _work_area(window):
    """Origin and size, in pixels, of the screen above the taskbar."""
    import ctypes

    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = (
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        )

    class MONITORINFO(ctypes.Structure):
        _fields_ = (
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        )

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    monitor = user32.MonitorFromWindow(int(window.wm_frame(), 16), 2)
    if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        area = info.rcWork
        return area.left, area.top, area.right - area.left, area.bottom - area.top
    return 0, 0, window.winfo_screenwidth(), window.winfo_screenheight()


def _frame_box(window):
    """Outer window rectangle, or None while it is still unrealized."""
    import ctypes

    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = (
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        )

    rect = RECT()
    if not user32.GetWindowRect(int(window.wm_frame(), 16), ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width < 50 or height < 50:
        return None
    return rect.left, rect.top, width, height


def _screen_margin(work_w, work_h):
    """Gap that keeps the frame a little inside the work area."""
    return max(40, int(round(work_w * 0.02))), max(40, int(round(work_h * 0.02)))


def _pixel_ratio(fig, window):
    """Physical pixels per inch of figure size, including display scaling."""
    ratio = getattr(fig.canvas, "device_pixel_ratio", 1.0) or 1.0
    if ratio > 1.01:
        return fig.dpi
    try:
        scaling = float(window.tk.call("tk", "scaling"))
        ratio = round(scaling / (96 / 72), 2)
    except (AttributeError, ValueError, tkinter.TclError):
        ratio = 1.0
    if ratio <= 0:
        ratio = 1.0
    return fig.dpi * ratio


def _frame_chrome(window, widget):
    """Extra width and height of the outer frame around the canvas."""
    box = _frame_box(window)
    if box is None:
        return None
    _left, _top, outer_w, outer_h = box
    chrome_w = outer_w - widget.winfo_width()
    chrome_h = outer_h - widget.winfo_height()
    if chrome_w < 8 or chrome_h < 30 or widget.winfo_width() < 50:
        return None
    return chrome_w, chrome_h


def _center_window(window):
    box = _frame_box(window)
    if box is None:
        return
    _x, _y, outer_w, outer_h = box
    left, top, work_w, work_h = _work_area(window)
    x = left + (work_w - outer_w) // 2
    y = top + (work_h - outer_h) // 2
    window.geometry(f"+{int(x)}+{int(y)}")


def _apply_fit(fig, design, tries=12):
    """Fit the designed figure inside the work area once the window is mapped."""
    window = fig.canvas.manager.window
    widget = fig.canvas._tkcanvas
    window.update_idletasks()
    px_per_in = _pixel_ratio(fig, window)
    design_w, design_h = design
    # The map handler scales the canvas up to the display. Measuring before
    # that lands reports the unscaled widget against a scaled frame.
    if widget.winfo_width() < design_w * px_per_in * 0.9:
        if tries > 0:
            window.after(50, _apply_fit, fig, design, tries - 1)
            return
        _center_window(window)
        return
    chrome = _frame_chrome(window, widget)
    if chrome is None:
        if tries > 0:
            window.after(50, _apply_fit, fig, design, tries - 1)
        else:
            _center_window(window)
        return
    chrome_w, chrome_h = chrome
    try:
        _left, _top, work_w, work_h = _work_area(window)
    except (OSError, AttributeError, ValueError):
        return
    margin_x, margin_y = _screen_margin(work_w, work_h)
    max_canvas_w = work_w - 2 * margin_x - chrome_w
    max_canvas_h = work_h - 2 * margin_y - chrome_h
    if max_canvas_w < 200 or max_canvas_h < 200 or px_per_in <= 0:
        return
    scale = min(
        1.0,
        max_canvas_w / (design_w * px_per_in),
        max_canvas_h / (design_h * px_per_in),
    )
    if scale < 0.999:
        fig.set_size_inches((design_w * scale, design_h * scale), forward=True)
        window.update_idletasks()
    _center_window(window)


def _fit_window(fig):
    """Open the window a little inside the screen.

    The frame's real pixel size, including the title bar, is known only
    after the window is mapped.
    """
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)
    widget = getattr(fig.canvas, "_tkcanvas", None)
    if window is None or widget is None:
        return
    design = tuple(float(size) for size in fig.get_size_inches())

    def place(_event):
        widget.unbind("<Map>", token)
        window.after(50, _apply_fit, fig, design)

    token = widget.bind("<Map>", place, add="+")


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


def _screen_dirs(axis, radiological):
    """World direction of screen-right and screen-up for one slice panel."""
    if axis == 0:
        # From the subject's right: anterior is screen-right, superior is up.
        return (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    # Radiological axial and coronal put subject right on the image left.
    lateral = -1.0 if radiological else 1.0
    if axis == 1:
        return (lateral, 0.0, 0.0), (0.0, 0.0, 1.0)
    return (lateral, 0.0, 0.0), (0.0, 1.0, 0.0)


def _display_geometry(affine, shape, axis, radiological):
    """Which voxel axis runs across and down the panel, and whether it is flipped."""
    u_axis, v_axis = [a for a in range(3) if a != axis]
    right, up = _screen_dirs(axis, radiological)
    du = affine[:3, u_axis]
    dv = affine[:3, v_axis]
    u_right = float(np.dot(du, right))
    u_up = float(np.dot(du, up))
    v_right = float(np.dot(dv, right))
    v_up = float(np.dot(dv, up))
    if abs(u_right) + abs(v_up) >= abs(v_right) + abs(u_up):
        col_axis, row_axis = u_axis, v_axis
        flip_col = u_right < 0
        flip_row = v_up > 0
    else:
        col_axis, row_axis = v_axis, u_axis
        flip_col = v_right < 0
        flip_row = u_up > 0
    return {
        "col_axis": col_axis,
        "row_axis": row_axis,
        "flip_col": flip_col,
        "flip_row": flip_row,
        "n_col": int(shape[col_axis]),
        "n_row": int(shape[row_axis]),
    }


def _present(data, axis, index, geom):
    """Slice turned so the panel's screen axes match _screen_dirs."""
    image = _slice(data, axis, index)
    u_axis = [a for a in range(3) if a != axis][0]
    shown = image.T if geom["col_axis"] == u_axis else image
    if geom["flip_col"]:
        shown = np.fliplr(shown)
    if geom["flip_row"]:
        shown = np.flipud(shown)
    return shown


def _display_coord(index, length, flipped):
    return (length - 1 - index) if flipped else index


def _place_side_letter(letter, radiological):
    """Park the L marker on the subject's left side of the panel."""
    if letter is None:
        return
    if radiological:
        letter.set_position((0.965, 0.50))
        letter.set_ha("right")
    else:
        letter.set_position((0.035, 0.50))
        letter.set_ha("left")


def _position_text(affine, axis, index):
    mm = _axis_mm(affine, axis, index)
    if mm is None:
        return str(int(index))
    return f"{'xyz'[axis]} = {mm:.0f} mm"


def _is_nifti(path):
    name = path.name.lower()
    return path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


def volume_choices(current):
    """NIfTI files next to this script, with the open file included."""
    current = Path(current).resolve()
    choices = sorted(
        (path.resolve() for path in ROOT.iterdir() if _is_nifti(path)),
        key=lambda path: path.name.lower(),
    )
    if current not in choices:
        choices.insert(0, current)
    return choices


class _VolumeMenu:
    """Closed filename button. The list opens upward over the 3D view."""

    def __init__(self, fig, rect, paths, current):
        self.fig = fig
        self.rect = rect
        self.paths = list(paths)
        self.current = Path(current)
        self.is_open = False
        self.popup = None
        self._rows = []
        self._hover = None
        self._over_button = False
        self.ax = fig.add_axes(rect, zorder=7)
        self.ax.set_navigate(False)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("#f6f6f6")
        for spine in self.ax.spines.values():
            spine.set_color("#b5b5b5")
            spine.set_linewidth(0.7)
        self.label = self.ax.text(
            0.03, 0.5, self.current.name,
            ha="left", va="center", fontsize=9, color="#222222",
            clip_on=True,
        )
        self.label.set_clip_path(Rectangle(
            (0.0, 0.0), 0.90, 1.0, transform=self.ax.transData,
        ))
        self._chevron = Polygon(
            [[0.942, 0.60], [0.978, 0.60], [0.960, 0.36]],
            closed=True, facecolor="#555555", edgecolor="none",
        )
        self.ax.add_patch(self._chevron)

    def set_label(self, text):
        self.label.set_text(text)

    def hover_button(self, over):
        if over == self._over_button:
            return False
        self._over_button = over
        self.ax.set_facecolor("#eaeaea" if over else "#f6f6f6")
        return True

    def _point_chevron(self, upward):
        if upward:
            points = [[0.942, 0.40], [0.978, 0.40], [0.960, 0.64]]
        else:
            points = [[0.942, 0.60], [0.978, 0.60], [0.960, 0.36]]
        self._chevron.set_xy(points)

    def open_menu(self):
        if self.is_open:
            return
        count = len(self.paths)
        x, y, width, height = self.rect
        room = max(height, 0.99 - (y + height))
        row_h = min(height, room / max(count, 1))
        popup_h = row_h * count
        self.popup = self.fig.add_axes([x, y + height, width, popup_h], zorder=8)
        self.popup.set_navigate(False)
        self.popup.set_xlim(0, 1)
        self.popup.set_ylim(count, 0)
        self.popup.set_xticks([])
        self.popup.set_yticks([])
        self.popup.set_facecolor("#ffffff")
        for spine in self.popup.spines.values():
            spine.set_color("#b5b5b5")
            spine.set_linewidth(0.7)
        self.popup.spines["bottom"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        self.ax.set_facecolor("#ffffff")
        self._rows = []
        for index, path in enumerate(self.paths):
            selected = path == self.current
            row = Rectangle(
                (0, index), 1, 1,
                facecolor="#e4edf8" if selected else "#ffffff",
                edgecolor="none",
            )
            self.popup.add_patch(row)
            weight = "medium" if selected else "normal"
            self.popup.text(
                0.03, index + 0.5, path.name,
                ha="left", va="center", fontsize=9, color="#1c1c1c",
                fontweight=weight, clip_on=True, zorder=2,
            )
            self._rows.append(row)
        self._point_chevron(True)
        self.is_open = True
        self._hover = None

    def close_menu(self):
        if not self.is_open:
            return
        self.popup.remove()
        self.popup = None
        self._rows = []
        self.is_open = False
        self._hover = None
        self._over_button = False
        self.ax.spines["top"].set_visible(True)
        self.ax.set_facecolor("#f6f6f6")
        self._point_chevron(False)

    def toggle(self):
        if self.is_open:
            self.close_menu()
        else:
            self.open_menu()

    def row_at(self, event):
        if not self.is_open or event.inaxes is not self.popup or event.ydata is None:
            return None
        row = int(np.floor(event.ydata))
        if 0 <= row < len(self.paths):
            return row
        return None

    def hover_row(self, row):
        if row == self._hover:
            return False
        self._hover = row
        for index, patch in enumerate(self._rows):
            selected = self.paths[index] == self.current
            if index == row:
                patch.set_facecolor("#d3e2f4" if selected else "#f0f0f0")
            else:
                patch.set_facecolor("#e4edf8" if selected else "#ffffff")
        return True


class _ConventionToggle:
    """Neurological and radiological segments above the 3D view."""

    LABELS = ("Neurological", "Radiological")

    def __init__(self, fig, rect, radiological=True):
        self.radiological = bool(radiological)
        self._hover = None
        self.ax = fig.add_axes(rect, zorder=6)
        self.ax.set_navigate(False)
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("#f6f6f6")
        self.ax.set_autoscale_on(False)
        for spine in self.ax.spines.values():
            spine.set_color("#b5b5b5")
            spine.set_linewidth(0.7)
        self._faces = []
        self._labels = []
        for index, name in enumerate(self.LABELS):
            face = Rectangle(
                (index, 0), 1, 1,
                facecolor="#f6f6f6", edgecolor="none", zorder=1,
            )
            self.ax.add_patch(face)
            label = self.ax.text(
                index + 0.5, 0.5, name,
                ha="center", va="center", fontsize=9, color="#1c1c1c",
                clip_on=True, zorder=2,
            )
            self._faces.append(face)
            self._labels.append(label)
        self._divider = self.ax.plot(
            [1, 1], [0, 1], color="#b5b5b5", lw=0.7,
            solid_capstyle="butt", zorder=3, clip_on=True,
        )[0]
        self._paint()

    def _paint(self):
        for index, (face, label) in enumerate(zip(self._faces, self._labels)):
            selected = (index == 1) == self.radiological
            if selected and index == self._hover:
                face.set_facecolor("#d3e2f4")
            elif selected:
                face.set_facecolor("#e4edf8")
            elif index == self._hover:
                face.set_facecolor("#eaeaea")
            else:
                face.set_facecolor("#f6f6f6")
            label.set_fontweight("medium" if selected else "normal")

    def side_at(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        return 0 if event.xdata < 1 else 1

    def hover(self, side):
        if side == self._hover:
            return False
        self._hover = side
        self._paint()
        return True

    def select(self, radiological):
        radiological = bool(radiological)
        if radiological == self.radiological:
            return False
        self.radiological = radiological
        self._paint()
        return True


def draw(path):
    path = Path(path).resolve()
    data, affine = load(path)
    lo, hi, fade, solid = display_levels(data)
    indices = index_at_world_zero(affine, data.shape)
    box = tissue_box(data, solid)
    state = {
        "path": path,
        "data": data,
        "affine": affine,
        "lo": lo,
        "hi": hi,
        "fade": fade,
        "solid": solid,
        "box": box,
        "radiological": True,
    }

    fig = plt.figure(figsize=(11.6, 11.2))
    fig_w, fig_h = fig.get_size_inches()
    _fit_window(fig)
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

    def panel_rect(n_col, n_row, col, row):
        px, py = cell_box(col, row)
        aspect = n_col / max(n_row, 1)
        avail_w = cell_w
        avail_h = cell_h - title_in
        if avail_w / max(avail_h, 0.1) > aspect:
            height = avail_h
            width = height * aspect
        else:
            width = avail_w
            height = width / aspect
        # Face the neighboring column instead of the window edge.
        x = px + cell_w - width if col == 0 else px
        return [
            x / fig_w,
            (py + cell_h - title_in - height) / fig_h,
            width / fig_w,
            height / fig_h,
        ]

    for axis, (col, row) in zip(PANEL_ORDER, ((0, 0), (1, 0), (0, 1))):
        geom = _display_geometry(affine, data.shape, axis, state["radiological"])
        ax = fig.add_axes(panel_rect(geom["n_col"], geom["n_row"], col, row))
        artist = ax.imshow(
            _present(data, axis, indices[axis], geom),
            cmap="gray",
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax.set_xlim(-0.5, geom["n_col"] - 0.5)
        ax.set_ylim(geom["n_row"] - 0.5, -0.5)
        x = _display_coord(indices[geom["col_axis"]], geom["n_col"], geom["flip_col"])
        y = _display_coord(indices[geom["row_axis"]], geom["n_row"], geom["flip_row"])
        v_line = ax.axvline(x, color=PLANES[geom["col_axis"]][2], lw=0.9)
        h_line = ax.axhline(y, color=PLANES[geom["row_axis"]][2], lw=0.9)
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
                0.965, 0.50, "L",
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=13, color="white", fontweight="medium",
                path_effects=[patheffects.withStroke(linewidth=2.4, foreground=(0, 0, 0, 0.7))],
                zorder=4,
            )
            _place_side_letter(letter, state["radiological"])
        panels.append({
            "axis": axis,
            "artist": artist,
            "v_line": v_line,
            "h_line": h_line,
            "axes": ax,
            "mm": mm,
            "letter": letter,
            "col": col,
            "row": row,
            **geom,
        })

    cx, cy = cell_box(1, 1)
    # Shorter sliders leave a row for the file menu. The 3D view is a
    # fraction of that leftover square, centered in it, so the planes sit
    # smaller in the cell and a little above the menu.
    slider_h = 0.15
    slider_gap = 0.04
    slider_bottom = 0.02
    menu_h = 0.24
    menu_gap = 0.05
    slider_block = 3 * slider_h + 2 * slider_gap
    plot_top = cy + cell_h - title_in
    plot_floor = cy + slider_bottom + slider_block + menu_gap + menu_h + 0.40
    # Inset so the z label and the slider names stay clear of the sagittal cell.
    inset = 0.34
    controls_w = min(cell_w - inset - 0.62, plot_top - plot_floor)
    side = controls_w * 0.72
    view_x = cx + inset + (controls_w - side) / 2
    view_y = plot_floor + (plot_top - plot_floor - side) / 2
    ax3 = fig.add_axes([
        view_x / fig_w,
        view_y / fig_h,
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
            (cy + slider_bottom + (2 - slot) * (slider_h + slider_gap)) / fig_h,
            controls_w / fig_w,
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
            handle_style={"size": 7},
        )
        slider.drawon = False
        slider.label.set_color(color)
        slider.label.set_fontsize(9)
        slider.valtext.set_color(color)
        slider.valtext.set_fontsize(9)
        sliders.append(slider)

    menu = _VolumeMenu(
        fig,
        [
            (cx + inset) / fig_w,
            (cy + slider_bottom + slider_block + menu_gap) / fig_h,
            controls_w / fig_w,
            menu_h / fig_h,
        ],
        volume_choices(path),
        path,
    )
    toggle_h = 0.24
    toggle_y = cy + cell_h - (title_in + toggle_h) / 2
    convention = _ConventionToggle(
        fig,
        [
            (cx + inset) / fig_w,
            toggle_y / fig_h,
            controls_w / fig_w,
            toggle_h / fig_h,
        ],
        radiological=state["radiological"],
    )

    shown = {"ready": False}

    def refresh_2d():
        data = state["data"]
        affine = state["affine"]
        current = [int(slider.val) for slider in sliders]
        for panel in panels:
            axis = panel["axis"]
            panel["artist"].set_data(_present(data, axis, current[axis], panel))
            x = _display_coord(current[panel["col_axis"]], panel["n_col"], panel["flip_col"])
            y = _display_coord(current[panel["row_axis"]], panel["n_row"], panel["flip_row"])
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
        data = state["data"]
        affine = state["affine"]
        lo = state["lo"]
        hi = state["hi"]
        fade = state["fade"]
        solid = state["solid"]
        box = state["box"]
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
        index = int(np.clip(round(index), 0, state["data"].shape[axis] - 1))
        if int(sliders[axis].val) == index:
            return
        sliders[axis].set_val(index)

    drag = {"axis": None, "y": 0.0, "index": 0, "height": 1.0}

    def sync_panel(panel, index):
        data = state["data"]
        affine = state["affine"]
        geom = _display_geometry(affine, data.shape, panel["axis"], state["radiological"])
        panel.update(geom)
        panel["axes"].set_position(panel_rect(geom["n_col"], geom["n_row"], panel["col"], panel["row"]))
        panel["artist"].set_data(_present(data, panel["axis"], index, geom))
        panel["artist"].set_extent((-0.5, geom["n_col"] - 0.5, geom["n_row"] - 0.5, -0.5))
        panel["axes"].set_xlim(-0.5, geom["n_col"] - 0.5)
        panel["axes"].set_ylim(geom["n_row"] - 0.5, -0.5)
        panel["v_line"].set_color(PLANES[geom["col_axis"]][2])
        panel["h_line"].set_color(PLANES[geom["row_axis"]][2])
        _place_side_letter(panel["letter"], state["radiological"])

    def apply_convention():
        current = [int(slider.val) for slider in sliders]
        for panel in panels:
            sync_panel(panel, current[panel["axis"]])
        was_ready = shown["ready"]
        shown["ready"] = False
        refresh_2d()
        shown["ready"] = was_ready

    def apply_volume(data, affine, new_path):
        lo, hi, fade, solid = display_levels(data)
        origin = index_at_world_zero(affine, data.shape)
        state.update(
            path=new_path,
            data=data,
            affine=affine,
            lo=lo,
            hi=hi,
            fade=fade,
            solid=solid,
            box=tissue_box(data, solid),
        )
        for panel in panels:
            sync_panel(panel, origin[panel["axis"]])
            panel["artist"].set_clim(lo, hi)
        low, high = volume_bounds(affine, data.shape)
        span = np.maximum(high - low, 1.0)
        pad = span * 0.03
        ax3.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax3.set_ylim(low[1] - pad[1], high[1] + pad[1])
        ax3.set_zlim(low[2] - pad[2], high[2] + pad[2])
        ax3.set_box_aspect(tuple(span), zoom=1.12)
        for slider, (_name, axis, _color) in zip(sliders, PLANES):
            slider.valmin = 0
            slider.valmax = max(data.shape[axis] - 1, 1)
            slider.ax.set_xlim(slider.valmin, slider.valmax)
            index = origin[axis]
            slider.eventson = False
            slider.set_val(index)
            slider.vline.set_xdata([index, index])
            slider.valinit = index
            slider.eventson = True
        drag["axis"] = None
        pending["dirty"] = False
        timer.stop()
        was_ready = shown["ready"]
        shown["ready"] = False
        refresh_2d()
        shown["ready"] = was_ready
        refresh_3d()
        fig.suptitle(new_path.name, fontsize=13)

    def pick(new_path):
        new_path = Path(new_path).resolve()
        menu.close_menu()
        if new_path == state["path"]:
            fig.canvas.draw_idle()
            return
        menu.set_label("Loading...")
        fig.canvas.draw()
        try:
            loaded, loaded_affine = load(new_path)
            apply_volume(loaded, loaded_affine, new_path)
        except Exception:
            menu.set_label(state["path"].name)
            fig.canvas.draw_idle()
            raise
        menu.current = new_path
        menu.set_label(new_path.name)

    def panel_at(inaxes):
        for panel in panels:
            if panel["axes"] is inaxes:
                return panel
        return None

    def on_scroll(event):
        if menu.is_open and event.inaxes in (menu.ax, menu.popup):
            return
        panel = panel_at(event.inaxes)
        if panel is None or not event.step:
            return
        step = 1 if event.step > 0 else -1
        set_plane(panel["axis"], int(sliders[panel["axis"]].val) + step)

    def on_press(event):
        if event.button == 1 and event.inaxes is convention.ax:
            closed = menu.is_open
            if closed:
                menu.close_menu()
            side = convention.side_at(event)
            if side is not None and convention.select(side == 1):
                state["radiological"] = convention.radiological
                apply_convention()
            elif closed:
                fig.canvas.draw_idle()
            return
        if event.button == 1 and event.inaxes is menu.ax:
            menu.toggle()
            fig.canvas.draw_idle()
            return
        if menu.is_open:
            row = menu.row_at(event) if event.button == 1 else None
            if row is not None:
                pick(menu.paths[row])
                return
            menu.close_menu()
            fig.canvas.draw_idle()
            if event.button != 1:
                return
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
        changed = False
        if menu.is_open:
            if menu.hover_row(menu.row_at(event)):
                changed = True
        elif menu.hover_button(event.inaxes is menu.ax):
            changed = True
        if convention.hover(convention.side_at(event)):
            changed = True
        if changed and drag["axis"] is None:
            fig.canvas.draw_idle()
        axis = drag["axis"]
        if axis is None or event.y is None:
            return
        # A drag the height of the panel walks the whole stack.
        span = max(state["data"].shape[axis] - 1, 1)
        delta = (event.y - drag["y"]) / drag["height"] * span
        set_plane(axis, drag["index"] + delta)

    def on_release(_event):
        if drag["axis"] is None and not pending["dirty"]:
            return
        drag["axis"] = None
        flush_3d()

    def on_draw(_event):
        shown["ready"] = True

    def on_key(event):
        if event.key == "escape" and menu.is_open:
            menu.close_menu()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", on_draw)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._slice_timer = timer
    # Sliders and the menu are only weakly referenced by the canvas.
    fig._slice_sliders = sliders
    fig._volume_menu = menu
    fig._convention_toggle = convention

    for slider, (_name, axis, _color) in zip(sliders, PLANES):
        slider.valtext.set_text(_position_text(affine, axis, indices[axis]))
    refresh_3d()
    pending["dirty"] = False
    fig.suptitle(path.name, fontsize=13)
    return fig


def main():
    if not PATH.exists():
        raise SystemExit(f"missing {PATH}")
    draw(PATH)
    plt.show()


if __name__ == "__main__":
    main()
