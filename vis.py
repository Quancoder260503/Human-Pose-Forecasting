"""
pose_visualizer.py

3D visualizer for human pose sequences produced by / fed into the
GaussianDiffusion Trainer (H36M-style skeleton).

Data conventions assumed (matching Trainer.train / Trainer.get_prediction):
  - Raw dataset poses:      [..., t_total, num_joints=17, 3]
  - Model input / output:   root joint (index 0) stripped out via
                             data[..., 1:, :], flattened to [..., t, 16*3]
This script accepts either 17-joint or 16-joint sequences. If 16 joints are
given, a zero root joint is prepended so the standard H36M skeleton can be
drawn (matches how the root is dropped upstream in Trainer.train / get_gt).

Usage:
    python pose_visualizer.py pred.npy --out pred.gif
    python pose_visualizer.py pred.pt --sample-index 0 --out pred.mp4 --fps 25
    python pose_visualizer.py gt.npy pred.npy --compare --out compare.gif

Accepted input formats: .npy, .pt/.pth (torch tensor), .csv (single sequence,
flattened [T, J*3]).
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive by default; overridden by --show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# H36M 17-joint skeleton definition
# 0 Hip(root) 1 RHip 2 RKnee 3 RFoot 4 LHip 5 LKnee 6 LFoot
# 7 Spine 8 Thorax 9 Neck/Nose 10 Head
# 11 LShoulder 12 LElbow 13 LWrist 14 RShoulder 15 RElbow 16 RWrist
# ---------------------------------------------------------------------------
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
LEFT_BONES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT_BONES = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}


# ---------------------------------------------------------------------------
# Loading / shape handling
# ---------------------------------------------------------------------------
def load_pose_array(path):
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".pt") or path.endswith(".pth"):
        import torch  # local import: only needed for torch checkpoint files
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    elif path.endswith(".csv"):
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file extension for {path}")
    return np.asarray(arr)


def select_sequence(arr, sample_index=None):
    """Collapse leading batch/sample dims down to a single sequence.

    Handles shapes coming out of get_prediction (which yields
    [b*sample_num, t_pred, nc] before rearrange, or [b, s, t, nc] after) as
    well as raw dataset batches [b, t, j, 3].
    """
    idx = sample_index if sample_index is not None else 0
    if arr.ndim == 5:      # [b, s, t, j, c]
        arr = arr[0, idx]
    elif arr.ndim == 4 and arr.shape[-1] == 3:  # [b, t, j, c]
        arr = arr[idx]
    elif arr.ndim == 4:    # [b, s, t, nc] flattened joints
        arr = arr[0, idx]
    elif arr.ndim == 3 and arr.shape[-1] != 3:  # [n, t, nc] flattened joints
        arr = arr[idx]
    # ndim == 3 with last dim 3, or ndim == 2 -> already a single sequence
    return arr


def to_TJC(arr, num_joints=None):
    """Coerce a sequence array to shape [T, J, 3]."""
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2:
        T = arr.shape[0]
        if num_joints is None:
            if arr.shape[1] % 3 != 0:
                raise ValueError(
                    "Cannot infer num_joints from flattened array; pass --num-joints"
                )
            num_joints = arr.shape[1] // 3
        return arr.reshape(T, num_joints, 3)
    raise ValueError(f"Cannot convert array of shape {arr.shape} to [T, J, 3]")


def maybe_add_root(seq):
    """Prepend a zero root joint if only 16 joints are present (root was
    stripped upstream via data[..., 1:, :], as in Trainer.train)."""
    T, J, C = seq.shape
    if J == 16:
        root = np.zeros((T, 1, C), dtype=seq.dtype)
        seq = np.concatenate([root, seq], axis=1)
    elif J != 17:
        raise ValueError(f"Expected 16 or 17 joints, got {J}")
    return seq


def center_sequence(seq):
    """Center every frame on its root joint for a stable camera."""
    return seq - seq[:, 0:1, :]


def compute_axis_limits(*seqs, pad=0.1):
    all_pts = np.concatenate([s.reshape(-1, 3) for s in seqs], axis=0)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    span = max((maxs - mins).max(), 1e-6)
    mid = (maxs + mins) / 2
    half = span / 2 * (1 + pad)
    return mid - half, mid + half


def prep(path, sample_index, num_joints):
    arr = load_pose_array(path)
    arr = select_sequence(arr, sample_index)
    seq = to_TJC(arr, num_joints=num_joints)
    seq = maybe_add_root(seq)
    return center_sequence(seq)


def prep_array(arr, sample_index=0, num_joints=None):
    """Same pipeline as prep(), but for an in-memory array/tensor instead of
    a file path. Use this to plug straight into Trainer without saving to
    disk first, e.g.:

        pred = self.get_prediction(data, extra['act'], sample_num=1,
                                    uncond=True).detach().cpu().numpy()
        seq = prep_array(pred, sample_index=0)   # -> [T, 17, 3], centered
        animate_pose_sequence(seq, out_path="pred.gif")

    `arr` may be a numpy array or a torch tensor of shape
    [batch_size, time_sequence, pose_dim] (pose_dim = 48 for 16 joints with
    root stripped, or 51 for the full 17-joint skeleton).
    """
    if hasattr(arr, "detach"):  # torch tensor, without hard-requiring torch
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    arr = select_sequence(arr, sample_index)
    seq = to_TJC(arr, num_joints=num_joints)
    seq = maybe_add_root(seq)
    return center_sequence(seq)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _init_skeleton_artists(ax, scatter_kwargs=None):
    scat = ax.scatter([], [], [], c="black", s=15, depthshade=True, **(scatter_kwargs or {}))
    lines = []
    for bone in H36M_BONES:
        color = "tab:blue" if bone in LEFT_BONES else ("tab:red" if bone in RIGHT_BONES else "gray")
        (line,) = ax.plot([], [], [], c=color, linewidth=2)
        lines.append(line)
    return scat, lines


def _update_skeleton_artists(scat, lines, pose):
    xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
    scat._offsets3d = (xs, ys, zs)
    for line, (j1, j2) in zip(lines, H36M_BONES):
        line.set_data([xs[j1], xs[j2]], [ys[j1], ys[j2]])
        line.set_3d_properties([zs[j1], zs[j2]])


def _style_ax(ax, lo, hi, elev, azim):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Single-sequence animation
# ---------------------------------------------------------------------------
def animate_pose_sequence(seq, out_path=None, fps=20, elev=15, azim=70, show=False, title=None):
    """seq: [T, J, 3] with J in {16, 17} (16 gets a zero root prepended)."""
    seq = center_sequence(maybe_add_root(seq))
    lo, hi = compute_axis_limits(seq)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _style_ax(ax, lo, hi, elev, azim)
    scat, lines = _init_skeleton_artists(ax)

    def update(i):
        _update_skeleton_artists(scat, lines, seq[i])
        ax.set_title(f"{title or 'Pose sequence'}  frame {i + 1}/{len(seq)}")
        return [scat, *lines]

    anim = FuncAnimation(fig, update, frames=len(seq), interval=1000 / fps, blit=False)
    _save_or_show(anim, fig, out_path, fps, show)
    return anim


# ---------------------------------------------------------------------------
# Side-by-side GT vs. prediction animation (e.g. get_gt vs get_prediction)
# ---------------------------------------------------------------------------
def animate_comparison(seq_a, seq_b, labels=("GT", "Pred"), out_path=None,
                        fps=20, elev=15, azim=70, show=False):
    seq_a = center_sequence(maybe_add_root(seq_a))
    seq_b = center_sequence(maybe_add_root(seq_b))
    T = min(len(seq_a), len(seq_b))
    seq_a, seq_b = seq_a[:T], seq_b[:T]
    lo, hi = compute_axis_limits(seq_a, seq_b)

    fig = plt.figure(figsize=(11, 6))
    ax_a = fig.add_subplot(121, projection="3d")
    ax_b = fig.add_subplot(122, projection="3d")
    for ax, label in ((ax_a, labels[0]), (ax_b, labels[1])):
        _style_ax(ax, lo, hi, elev, azim)
        ax.set_title(label)
    scat_a, lines_a = _init_skeleton_artists(ax_a)
    scat_b, lines_b = _init_skeleton_artists(ax_b)

    def update(i):
        _update_skeleton_artists(scat_a, lines_a, seq_a[i])
        _update_skeleton_artists(scat_b, lines_b, seq_b[i])
        fig.suptitle(f"frame {i + 1}/{T}")
        return [scat_a, *lines_a, scat_b, *lines_b]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    _save_or_show(anim, fig, out_path, fps, show)
    return anim


# ---------------------------------------------------------------------------
# Static grid of samples at one frame (useful for eyeballing diversity,
# e.g. across cfg.eval_sample_num draws from get_prediction)
# ---------------------------------------------------------------------------
def plot_sample_grid(seqs, frame_idx=-1, out_path="sample_grid.png", elev=15, azim=70,
                      ncols=4, titles=None):
    """seqs: list of [T, J, 3] arrays (all same T), e.g. multiple diffusion
    samples for one input. Plots the pose at `frame_idx` for each sample."""
    n = len(seqs)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    seqs = [center_sequence(maybe_add_root(s)) for s in seqs]
    lo, hi = compute_axis_limits(*seqs)

    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for i, seq in enumerate(seqs):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _style_ax(ax, lo, hi, elev, azim)
        scat, lines = _init_skeleton_artists(ax)
        _update_skeleton_artists(scat, lines, seq[frame_idx])
        ax.set_title(titles[i] if titles else f"sample {i}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved sample grid to {out_path}")


# ---------------------------------------------------------------------------
def _save_or_show(anim, fig, out_path, fps, show):
    if out_path:
        if out_path.endswith(".gif"):
            anim.save(out_path, writer=PillowWriter(fps=fps))
        elif out_path.endswith(".mp4"):
            anim.save(out_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("out_path must end in .gif or .mp4")
        print(f"Saved animation to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize 3D human pose sequences.")
    parser.add_argument("inputs", nargs="+", help="One path to animate, or two (--compare) for GT vs. pred")
    parser.add_argument("--compare", action="store_true", help="Show two sequences side by side")
    parser.add_argument("--sample-index", type=int, default=None,
                         help="Which sample/batch element to pick if the array has extra leading dims")
    parser.add_argument("--num-joints", type=int, default=None,
                         help="Needed only if input is flattened [T, J*3]")
    parser.add_argument("--out", type=str, default="pose_sequence.gif", help="Output path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--elev", type=float, default=15)
    parser.add_argument("--azim", type=float, default=70)
    parser.add_argument("--show", action="store_true", help="Also display an interactive window")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if args.compare:
        if len(args.inputs) != 2:
            parser.error("--compare requires exactly two input paths (GT, Pred)")
        seq_a = prep(args.inputs[0], args.sample_index, args.num_joints)
        seq_b = prep(args.inputs[1], args.sample_index, args.num_joints)
        animate_comparison(seq_a, seq_b, out_path=args.out, fps=args.fps,
                            elev=args.elev, azim=args.azim, show=args.show)
    else:
        seq = prep(args.inputs[0], args.sample_index, args.num_joints)
        animate_pose_sequence(seq, out_path=args.out, fps=args.fps, elev=args.elev,
                               azim=args.azim, show=args.show, title=args.title)


if __name__ == "__main__":
    main()