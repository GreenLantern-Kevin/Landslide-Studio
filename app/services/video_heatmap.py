"""Export local sparse-flow speed maps from recorded samples, without rerunning flow.

The NPZ uses processed-frame coordinates. NaN means unobserved, never zero motion.
PNG figures use publication-size axes and a horizontal rainbow colorbar at 300 dpi.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


NOTES = [
    "散点图显示原始稀疏轨迹终点速度；连续云图显示其局部高斯加权平滑，不是稠密光流或全坡面实测速度。",
    "支持范围限于半径内且位于 ROI 的像素；无样本或无支持位置为 NaN，不代表静止。",
    "速度单位为处理像素/秒（px/s），未经空间标定不能解释为 m/s。CSV/NPZ 仍以左上角为原点；绘图纵轴向上，显示 y=height-1-y_px，背景同步对应。",
    "叠加背景来自已保存四宫格视频左上格，防抖任务采用该格的已对齐坐标；保留原视频面板文字。",
]


def _finite(value, label, low=None, high=None, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} 必须为有限数值") from None
    if not math.isfinite(number) or (low is not None and number < low) or (high is not None and number > high):
        raise ValueError(f"{label} 超出允许范围")
    if integer and number != int(number):
        raise ValueError(f"{label} 必须为整数")
    return int(number) if integer else number


def _safe_file(job_dir, name):
    path = job_dir / name
    if path.is_symlink() or path.resolve().parent != job_dir:
        raise ValueError("任务文件路径不安全")
    if not path.is_file():
        raise ValueError(f"任务缺少 {name}，请先完成稀疏光流分析")
    return path


def _load_samples(path, frame_index, width, height, roi, qualified_only, speed_mode):
    samples, rejected = [], 0
    speed_key = "speed_px_per_s" if speed_mode == "instantaneous" else "net_window_speed_px_per_s"
    required = {"frame_index", "track_id", "x_px", "y_px", "qualified", speed_key}
    x0, y0, x1, y1 = roi
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("tracks.csv 缺少轨迹速度字段，无法导出热图")
        for row in reader:
            try:
                row_frame = _finite(row.get("frame_index"), "frame_index", 0, integer=True)
            except ValueError:
                continue
            if row_frame != frame_index:
                continue
            # Frame zero is a reference frame; fabricated frame-zero rows are not measurements.
            if frame_index == 0:
                rejected += 1
                continue
            try:
                x = _finite(row.get("x_px"), "x")
                y = _finite(row.get("y_px"), "y")
                speed = _finite(row.get(speed_key), "speed", 0, float(np.finfo(np.float32).max))
                qualified = _finite(row.get("qualified"), "qualified", 0, 1, integer=True)
                track_id = _finite(row.get("track_id"), "track_id", 0, integer=True)
                if not (0 <= x < width and 0 <= y < height and x0 <= x < x1 and y0 <= y < y1):
                    raise ValueError("outside ROI")
                if qualified_only and not qualified:
                    continue
                if speed_mode == "window":
                    intervals = _finite(row.get("window_intervals"), "window_intervals", 1, integer=True)
                else:
                    intervals = 1
            except ValueError:
                rejected += 1
                continue
            samples.append(dict(track_id=track_id, x_px=x, y_px=y, selected_speed_px_per_s=speed,
                                qualified=qualified, window_intervals=intervals))
    return samples, rejected


def _smooth(samples, width, height, roi, radius):
    sums = np.zeros((height, width), np.float64)
    weights = np.zeros((height, width), np.float64)
    x0, y0, x1, y1 = roi
    sigma = radius / 3.0
    for sample in samples:
        x, y = sample["x_px"], sample["y_px"]
        left, right = max(x0, math.ceil(x - radius)), min(x1, math.floor(x + radius) + 1)
        top, bottom = max(y0, math.ceil(y - radius)), min(y1, math.floor(y + radius) + 1)
        xx = np.arange(left, right, dtype=np.float64) - x
        yy = np.arange(top, bottom, dtype=np.float64) - y
        distance_sq = yy[:, None] ** 2 + xx[None, :] ** 2
        kernel = np.exp(-distance_sq / (2 * sigma * sigma))
        kernel[distance_sq > radius * radius] = 0
        weights[top:bottom, left:right] += kernel
        sums[top:bottom, left:right] += kernel * sample["selected_speed_px_per_s"]
    support = weights > 0
    speed = np.full((height, width), np.nan, np.float32)
    speed[support] = (sums[support] / weights[support]).astype(np.float32)
    return speed, support


def _background(path, frame_index, width, height):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("无法打开已保存的分析视频")
        # OpenCV/FFmpeg seeks and decodes to the requested frame for this saved CFR MP4.
        if frame_index and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError("分析视频无法定位至所选帧")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("分析视频中无法读取所选帧")
        if frame.shape[:2] != (height * 2, width * 2):
            raise ValueError("分析视频尺寸与摘要中的四宫格布局不符，不能安全对齐热图")
        return cv2.cvtColor(frame[:height, :width], cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _figure(path, background, speed, samples, vmax, frame_index, fps, mode,
            plot_style, point_size, opacity, overlay):
    """Render saved measurements only, with consistent geometry and color limits.

    Direct Figure/Agg use avoids pyplot windows and its shared figure registry.
    The raster extent and H-1-y scatter transform align pixel centers exactly.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    height, width = speed.shape
    aspect = width / height
    axes_width = min(10.0, 8.0 * aspect)
    axes_height = axes_width / aspect
    fig_width = max(6.0, axes_width + 1.4)
    fig_height = max(3.8, axes_height + 2.15)
    left = (fig_width - axes_width) / 2
    fig = Figure(figsize=(fig_width, fig_height), dpi=300, facecolor='white')
    FigureCanvasAgg(fig)
    font = 'DejaVu Serif'
    cmap = colormaps['rainbow'].with_extremes(bad=(1, 1, 1, 0))
    norm = Normalize(vmin=0, vmax=vmax, clip=True)
    extent = (-0.5, width - 0.5, -0.5, height - 0.5)
    try:
        ax = fig.add_axes([left / fig_width, 1.55 / fig_height,
                           axes_width / fig_width, axes_height / fig_height])
        ax.set_facecolor('white')
        if overlay:
            ax.imshow(background, origin='upper', extent=extent, interpolation='bilinear')
        if plot_style == 'scatter':
            # Radius and the smoothed NPZ never replace the actual scatter values.
            ax.scatter([s['x_px'] for s in samples],
                       [height - 1 - s['y_px'] for s in samples],
                       c=[s['selected_speed_px_per_s'] for s in samples],
                       cmap=cmap, norm=norm, s=point_size, marker='o',
                       edgecolors='none', linewidths=0,
                       alpha=opacity if overlay else 1.0)
        else:
            # Keep the unsupported pixels transparent, rather than inventing a
            # full-field interpolation across unsampled parts of the hillside.
            ax.imshow(np.ma.masked_invalid(speed), origin='upper', extent=extent,
                      cmap=cmap, norm=norm, interpolation='bilinear',
                      interpolation_stage='rgba', alpha=opacity if overlay else 1.0)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (px)', fontsize=15, fontfamily=font, labelpad=7)
        ax.set_ylabel('Y (px)', fontsize=15, fontfamily=font, labelpad=7)
        ax.set_title(f'T = {frame_index / fps:g} s', loc='left', fontsize=17,
                     fontfamily=font, pad=13)
        ax.set_title(f'Frame {frame_index + 1}', loc='right', fontsize=12,
                     fontfamily=font, color='#444444', pad=13)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        ax.tick_params(direction='out', length=5, width=1.1, labelsize=13, pad=5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        bar_width = max(3.8, axes_width)
        cax = fig.add_axes([(fig_width - bar_width) / (2 * fig_width),
                            .55 / fig_height, bar_width / fig_width, .23 / fig_height])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           orientation='horizontal')
        bar.set_label('Speed (px/s)', fontsize=17, fontfamily=font, labelpad=8)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))
        bar.locator = MaxNLocator(nbins=6, min_n_ticks=3)
        bar.formatter = formatter
        bar.update_ticks()
        bar.ax.tick_params(labelsize=14, length=5, width=1.0, pad=5)
        for label in bar.ax.get_xticklabels():
            label.set_fontfamily(font)
        bar.outline.set_linewidth(1.1)
        if not samples:
            ax.text(.5, .5, 'No valid speed samples', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, fontfamily=font,
                    bbox=dict(facecolor='white', edgecolor='#aaaaaa', alpha=.95, pad=9))
        # Scientific definitions and sample counts stay in the companion JSON,
        # leaving the figure suitable for a paper without tiny debug footnotes.
        # Include every axis/colourbar label even for large coordinate values.
        fig.savefig(path, dpi=300, format='png', facecolor='white',
                    bbox_inches='tight', pad_inches=.18)
    finally:
        fig.clear()


def export_sparse_heatmap(job_dir: Path, frame_index: int, params: dict) -> dict:
    """Export five new files; frame_index is zero-based. Does not mutate summary.json.

    params: speed_mode=instantaneous/window, qualified_only=True, radius=24
    (3..200 processed pixels), vmax=None (auto) or positive, opacity=.6,
    plot_style=scatter/smooth, point_size=12 (1..100 points squared).
    """
    job_dir = Path(job_dir).resolve()
    if not job_dir.is_dir() or not isinstance(params, dict):
        raise ValueError("任务目录或热图参数无效")
    mode = params.get("speed_mode", "instantaneous")
    if mode not in {"instantaneous", "window"}:
        raise ValueError("speed_mode 必须为 instantaneous 或 window")
    plot_style = params.get('plot_style', 'scatter')
    if plot_style not in {'scatter', 'smooth'}:
        raise ValueError('plot_style 必须为 scatter 或 smooth')
    point_size = _finite(params.get('point_size', 12), 'point_size', 1, 100)
    qualified_only = params.get("qualified_only", True)
    if not isinstance(qualified_only, bool):
        raise ValueError("qualified_only 必须为布尔值")
    radius = _finite(params.get("radius", 24), "radius", 3, 200, integer=True)
    opacity = _finite(params.get("opacity", .6), "opacity", 0, 1)
    manual_vmax = params.get("vmax")
    if manual_vmax is not None:
        manual_vmax = _finite(manual_vmax, "vmax", np.finfo(np.float32).tiny, float(np.finfo(np.float32).max))
    summary = json.loads(_safe_file(job_dir, "summary.json").read_text(encoding="utf-8-sig"))
    source = summary.get("metrics", {})
    if summary.get("algorithm", source.get("algorithm")) != "sparse":
        raise ValueError("速度轨迹热图仅支持已完成的稀疏光流任务")
    width = _finite(source.get("width"), "width", 1, 32768, True)
    height = _finite(source.get("height"), "height", 1, 32768, True)
    if width * height > 40_000_000:
        raise ValueError("热图尺寸超过 4000 万像素，请先降低视频处理分辨率")
    frames = _finite(source.get("frames"), "frames", 1, integer=True)
    fps = _finite(source.get("fps"), "fps", .000001, 10000)
    frame_index = _finite(frame_index, "frame_index", 0, frames - 1, True)
    roi = source.get("roi_processed_pixels", [0, 0, width, height])
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("任务 ROI 信息无效")
    roi = [_finite(v, "ROI", 0, integer=True) for v in roi]
    if not (roi[0] < roi[2] <= width and roi[1] < roi[3] <= height):
        raise ValueError("任务 ROI 超出视频尺寸")
    samples, rejected = _load_samples(_safe_file(job_dir, "tracks.csv"), frame_index, width, height, roi, qualified_only, mode)
    background = _background(_safe_file(job_dir, "analysis.mp4"), frame_index, width, height)
    speed, support = _smooth(samples, width, height, roi, radius)
    sample_speeds = [sample["selected_speed_px_per_s"] for sample in samples]
    sample_max = max(sample_speeds) if samples else None
    vmax = manual_vmax if manual_vmax is not None else (sample_max if sample_max and sample_max > 0 else 1.0)
    prefix = f"speed_frame_{frame_index + 1:06d}_{uuid4().hex[:12]}"
    files = [
        dict(name=f"{prefix}_overlay.png", label=f"第 {frame_index + 1} 帧速度叠加图", kind="image"),
        dict(name=f"{prefix}_cloud.png", label=f"第 {frame_index + 1} 帧速度" + ('散点图' if plot_style == 'scatter' else '云图'), kind="image"),
        dict(name=f"{prefix}_samples.csv", label="热图有效样本 CSV", kind="data"),
        dict(name=f"{prefix}_field.npz", label="热图数值场 NPZ（NaN 为无数据）", kind="data"),
        dict(name=f"{prefix}_metadata.json", label="热图参数与说明 JSON", kind="data"),
    ]
    metrics = dict(frame_index=frame_index, frame_number=frame_index + 1, time_sec=frame_index / fps,
                   speed_mode=mode, qualified_only=qualified_only, radius_px=radius, gaussian_sigma_px=radius / 3,
                   opacity=opacity, colorbar_min=0, colorbar_max=vmax, colorbar_mode="manual" if manual_vmax is not None else "auto",
                   sample_count=len(samples), rejected_sample_count=rejected, support_pixel_count=int(support.sum()),
                   sample_min_speed_px_per_s=min(sample_speeds) if samples else None,
                   sample_max_speed_px_per_s=sample_max, width=width, height=height, fps=fps,
                   roi_processed_pixels=roi, has_data=bool(samples), unit="processed_px/s",
                   clipped_sample_count=sum(s > vmax for s in sample_speeds),
                   plot_style=plot_style, point_size=point_size, figure_dpi=300, colormap='rainbow',
                   plot_coordinates='x=x_px; y=height-1-y_px; y increases upwards',
                   numeric_field='local Gaussian smoothing; CSV retains unsmoothed samples')
    notes = NOTES + ["瞬时速度取相邻帧位移/帧间隔。" if mode == "instantaneous" else "窗口速度取轨迹首尾净位移/实际窗口时长，在当前终点表示；不是累计路程平均速度。"]
    if not samples:
        notes = notes + ["当前帧没有符合筛选条件的有效速度样本；图像标记无数据，数值场全部为 NaN。"]
    if not sample_max:
        notes = notes + ["自动色标在无数据或速度全零时使用 0–1 px/s，仅供显示；不表示测得 1 px/s。"] if manual_vmax is None else notes
    result = dict(files=files, metrics=metrics, notes=notes)
    created = []
    try:
        for index in range(2):
            path = job_dir / files[index]["name"]
            created.append(path)
            _figure(path, background, speed, samples, vmax, frame_index, fps, mode,
                    plot_style, point_size, opacity, overlay=index == 0)
        path = job_dir / files[2]["name"]
        created.append(path)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            fieldnames = ["frame_index", "frame_number", "time_sec", "speed_mode", "track_id", "x_px", "y_px", "selected_speed_px_per_s", "qualified", "window_intervals"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                writer.writerow(dict(frame_index=frame_index, frame_number=frame_index + 1, time_sec=frame_index / fps, speed_mode=mode, **sample))
        path = job_dir / files[3]["name"]
        created.append(path)
        np.savez_compressed(path, speed=speed, support_mask=support, frame_index=np.int64(frame_index),
                            time_sec=np.float64(frame_index / fps), roi_processed_pixels=np.asarray(roi, np.int64),
                            unit=np.asarray("processed_px/s"), coordinates=np.asarray("origin=top-left; x=right; y=down; shape=(height,width)"),
                            x_px=np.arange(width, dtype=np.int32), y_px=np.arange(height, dtype=np.int32), speed_mode=np.asarray(mode))
        path = job_dir / files[4]["name"]
        created.append(path)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return result
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
