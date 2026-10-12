"""Headless adaptation of GreenLantern-Kevin's optical-flow research scripts.

Source repository: https://github.com/GreenLantern-Kevin/Optical-Flow-Seismic-Rock-Avalanche-Detection
Source commit: bc39d90061a10e880927087f510c3e16b41230dd
Adapted paths: sparseOpticalFlow.py (forward/backward LK, re-detection, trajectory
filtering, flow energy), denseOpticalFlow.py (Farneback, HSV/H/V visualizations),
Landslide_Monitor_Warning.py (background SIFT/FLANN/RANSAC stabilization, ROI
thresholding). This is an explicit adaptation, not an import of the GUI scripts.
Parameter reference updated from the user's desktop sparseOpticalFlow.py and
denseOpticalFlow.py on 2026-09-25. Native processing is the default; four panels
retain their processing size and form an exact 2W-by-2H output.

Changes: injectable paths and parameters; no GUI; metadata FPS; proper interval
counts; first-frame retention; finite/empty-feature handling; normalized ROI;
stabilization failure exclusion; H264 output; structured CSV/JSON and track CSV.
Unlike the reference's per-detection maxCorners, max_corners bounds simultaneous
tracks. Display gains and line widths never change measured displacement/speed.
All motion values refer to PROCESSED pixels; no geophysical speed is inferred.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

SOURCE = {
    "repository": "https://github.com/GreenLantern-Kevin/Optical-Flow-Seismic-Rock-Avalanche-Detection",
    "commit": "bc39d90061a10e880927087f510c3e16b41230dd",
    "adapter": "headless_research_adaptation_v2",
    "parameter_reference": "user-provided desktop sparseOpticalFlow.py / denseOpticalFlow.py, 2026-09-25",
}


def _number(params, key, default, low, high, integer=False):
    value = float(params.get(key, default))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} 必须介于 {low} 和 {high} 之间")
    if integer and value != int(value):
        raise ValueError(f"{key} 必须是整数")
    return int(value) if integer else value


def _roi(params, width, height):
    value = params.get("roi") or {"x": 0, "y": 0, "width": 1, "height": 1}
    if not isinstance(value, dict):
        raise ValueError("ROI 应为 {x,y,width,height}，取值为 0–1 的归一化坐标")
    x, y, rw, rh = (float(value.get(k, d)) for k, d in [("x", 0), ("y", 0), ("width", 1), ("height", 1)])
    if not all(math.isfinite(v) for v in [x, y, rw, rh]) or x < 0 or y < 0 or rw <= 0 or rh <= 0 or x + rw > 1.000001 or y + rh > 1.000001:
        raise ValueError("ROI 必须位于影像内；坐标、宽和高均使用 0–1 归一化值")
    x0, y0 = int(x * width), int(y * height)
    x1, y1 = min(width, math.ceil((x + rw) * width)), min(height, math.ceil((y + rh) * height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("缩放后的 ROI 宽和高均至少为 8 像素")
    mask = np.zeros((height, width), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (x0, y0, x1, y1)


class _Stabilizer:
    """Reference-to-current SIFT matches, current-to-reference RANSAC warp."""

    def __init__(self, gray, roi_mask):
        self.height, self.width = gray.shape
        background = cv2.bitwise_not(roi_mask)
        if np.count_nonzero(background) < 100:
            raise ValueError("防抖需要 ROI 外的静态背景；请设置小于全图的 ROI 或关闭防抖")
        self.detector = cv2.SIFT_create()
        self.keypoints, self.descriptors = self.detector.detectAndCompute(gray, background)
        if self.descriptors is None or len(self.keypoints) < 10:
            raise ValueError("首帧 ROI 外不足 10 个 SIFT 特征点，无法可靠防抖；请调整 ROI 或关闭防抖")
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def apply(self, frame, gray):
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 10:
            return frame, gray, None, "failed_no_features"
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
        if len(matches) < 10:
            return frame, gray, None, "failed_few_matches"
        ref = np.float32([self.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(curr, ref, cv2.RANSAC, 5.0)
        if matrix is None or not np.isfinite(matrix).all() or inliers is None or int(inliers.sum()) < 10:
            return frame, gray, None, "failed_homography"
        size = (self.width, self.height)
        valid = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
        # Exclude warp borders, whose artificial contrast can create spurious flow.
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
        return cv2.warpPerspective(frame, matrix, size), cv2.warpPerspective(gray, matrix, size), valid, "aligned"


def _panel(image, label):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 23), (15, 24, 35), -1)
    cv2.putText(image, label, (7, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 245, 250), 1, cv2.LINE_AA)
    return image


def _dense_panels(frame, gray, flow, bounds, threshold, *, vector_step=16,
                  vector_scale=3, line_thickness=2, motion_gain=20):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.nan_to_num(magnitude, nan=0, posinf=0, neginf=0)
    hsv = np.zeros((*gray.shape, 3), np.uint8)
    # Preserve the reference angle convention without its H*5 wrapping artifact.
    hsv[..., 0] = ((angle + np.pi) * (90 / np.pi) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    # Clip AFTER applying the display gain: uint8 conversion must never wrap.
    hsv[..., 2] = np.clip(magnitude * motion_gain, 0, 255).astype(np.uint8)
    hsv[magnitude <= threshold, 2] = 0
    heat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    arrows = frame.copy()
    step = vector_step
    for y in range(step // 2, gray.shape[0], step):
        for x in range(step // 2, gray.shape[1], step):
            dx, dy = flow[y, x]
            if np.isfinite([dx, dy]).all() and magnitude[y, x] > threshold:
                # Bound drawing coordinates only; the measured flow is untouched.
                end = (round(np.clip(x + float(dx) * vector_scale, -gray.shape[1], gray.shape[1] * 2)),
                       round(np.clip(y + float(dy) * vector_scale, -gray.shape[0], gray.shape[0] * 2)))
                cv2.arrowedLine(arrows, (x, y), end, (90, 230, 85), line_thickness,
                                line_type=cv2.LINE_AA, tipLength=0.3)
    x0, y0, x1, y1 = bounds
    cv2.rectangle(arrows, (x0, y0), (x1 - 1, y1 - 1), (45, 210, 230), 2)
    cv2.rectangle(heat, (x0, y0), (x1 - 1, y1 - 1), (45, 210, 230), 2)
    return np.vstack((np.hstack((_panel(arrows, f"Dense vectors (arrows x{vector_scale:g})"), _panel(heat, "HSV motion: display enhancement"))),
                      np.hstack((_panel(hsv[..., 0], "H / direction"), _panel(hsv[..., 2], f"V / magnitude: px/frame x{motion_gain:g}")))))


def run_video(video_path: Path, out_dir: Path, params: dict, config: dict, progress: Callable[[float, str], None]) -> dict:
    """Analyze a video. result['files'] lists {name,label,kind}, relative to out_dir.

    params: algorithm ('sparse'/'dense'), stabilize (bool), max_width (0 = native,
    otherwise 64..3840, never upscale), roi ({x,y,width,height}, normalized),
    noise_threshold (processed px/frame, default .5), max_corners (10000 active
    tracks), quality_level (.3), min_distance (7), block_size (odd, 7),
    reseed_radius (5), track_thickness (2), trajectory_len (30), detect_interval
    (1), min_track_displacement (10 processed pixels), fb_threshold (1 processed
    pixel), fps_override (optional). LK uses 15/2/10/.03 as in the user's script.
    Dense parameters: dense_pyr_scale (.5), dense_levels (3), dense_winsize (15),
    dense_iterations (3), dense_poly_n (5), dense_poly_sigma (1.2).
    Drawing only: dense_vector_step (16), dense_vector_scale (3),
    dense_line_thickness (2), dense_motion_gain (20).
    config['_root'] is accepted for consistency; no global paths are used.
    """
    del config
    video_path, out_dir = Path(video_path), Path(out_dir)
    if not video_path.is_file():
        raise ValueError("找不到上传的视频文件")
    algorithm = params.get("algorithm", "sparse")
    if algorithm not in {"sparse", "dense"}:
        raise ValueError("algorithm 只能为 sparse 或 dense")
    stabilize = params.get("stabilize", False)
    if not isinstance(stabilize, bool):
        raise ValueError("stabilize 必须是布尔值")
    max_width = _number(params, "max_width", 0, 0, 3840, True)
    if 0 < max_width < 64:
        raise ValueError("max_width 取 0 表示保持原尺寸，或取 64–3840 表示限制宽度")
    threshold = _number(params, "noise_threshold", 0.5, 0, 100)
    max_corners = _number(params, "max_corners", 10000, 10, 10000, True)
    quality = _number(params, "quality_level", 0.3, 0.001, 0.99)
    min_distance = _number(params, "min_distance", 7, 1, 100)
    block_size = _number(params, "block_size", 7, 3, 31, True)
    if block_size % 2 == 0:
        raise ValueError("block_size 必须为 3–31 之间的奇数")
    reseed_radius = _number(params, "reseed_radius", 5, 0, 100, True)
    track_thickness = _number(params, "track_thickness", 2, 1, 10, True)
    trajectory_len = _number(params, "trajectory_len", 30, 2, 200, True)
    detect_interval = _number(params, "detect_interval", 1, 1, 100, True)
    min_displacement = _number(params, "min_track_displacement", 10, 0, 10000)
    fb_threshold = _number(params, "fb_threshold", 1, 0.05, 10)
    dense_params = dict(
        pyr_scale=_number(params, "dense_pyr_scale", 0.5, 0.05, 0.95),
        levels=_number(params, "dense_levels", 3, 1, 8, True),
        winsize=_number(params, "dense_winsize", 15, 3, 101, True),
        iterations=_number(params, "dense_iterations", 3, 1, 20, True),
        poly_n=_number(params, "dense_poly_n", 5, 5, 7, True),
        poly_sigma=_number(params, "dense_poly_sigma", 1.2, 0.1, 3),
        flags=0,
    )
    if dense_params["winsize"] % 2 == 0:
        raise ValueError("dense_winsize 必须为 3–101 之间的奇数")
    if dense_params["poly_n"] not in {5, 7}:
        raise ValueError("dense_poly_n 只能为 5 或 7")
    dense_visual = dict(
        vector_step=_number(params, "dense_vector_step", 16, 4, 128, True),
        vector_scale=_number(params, "dense_vector_scale", 3, 0.1, 50),
        line_thickness=_number(params, "dense_line_thickness", 2, 1, 10, True),
        motion_gain=_number(params, "dense_motion_gain", 20, 0.1, 1000),
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("无法打开视频；请尝试 H.264 编码的 MP4 或常见 AVI 文件")
    writer = None
    warnings = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if params.get("fps_override") is not None:
            fps = _number(params, "fps_override", 25, 0.1, 1000)
            warnings.append("时间轴使用手动指定的帧率。")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("无法读取有效视频帧率，请提供 fps_override 后重试")
        raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(raw_total) if math.isfinite(raw_total) and raw_total > 0 else 0
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("视频中没有可解码的帧")
        original_height, original_width = frame.shape[:2]
        ratio = min(1.0, max_width / original_width) if max_width else 1.0
        width, height = max(8, round(original_width * ratio)), max(8, round(original_height * ratio))
        if original_width < 8 or original_height < 8:
            raise ValueError("视频宽高至少为 8 像素")
        roi_mask, bounds = _roi(params, width, height)
        if ratio != 1:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stabilizer = _Stabilizer(prev_gray, roi_mask) if stabilize else None
        out_dir.mkdir(parents=True, exist_ok=True)
        video_file, csv_file, summary_file, preview_file = "analysis.mp4", "motion.csv", "summary.json", "preview.jpg"
        tracks_file = "tracks.csv"
        writer = imageio_ffmpeg.write_frames(str(out_dir / video_file), (width * 2, height * 2), fps=fps,
                    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7, macro_block_size=1,
                    output_params=["-movflags", "+faststart"], ffmpeg_log_level="error")
        writer.send(None)
        row_fields = ["frame_index", "time_sec", "measurement_valid", "status", "stabilization", "sample_count", "moving_count", "moving_fraction",
                      "mean_displacement_px_per_frame", "max_displacement_px_per_frame", "mean_speed_px_per_s", "max_speed_px_per_s",
                      "flow_energy_px2_per_frame2", "qualified_tracks"]
        track_fields = ["frame_index", "time_sec", "track_id", "x_px", "y_px", "dx_px", "dy_px", "speed_px_per_s", "window_intervals", "net_window_speed_px_per_s", "qualified"]
        trajectories, next_id = [], 0
        index = 0
        measured_frames = 0
        max_speed = total_mean_speed = total_energy = 0.0
        failed_stabilization = 0
        previous_valid = np.full((height, width), 255, np.uint8)
        previous_aligned = True
        preview_written = False
        lk = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        progress(1, "正在读取视频并初始化光流分析")
        with (out_dir / csv_file).open("w", newline="", encoding="utf-8-sig") as output, (out_dir / tracks_file).open("w", newline="", encoding="utf-8-sig") as track_output:
            rows = csv.DictWriter(output, fieldnames=row_fields)
            rows.writeheader()
            track_rows = csv.DictWriter(track_output, fieldnames=track_fields)
            track_rows.writeheader()
            while True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                valid = np.full((height, width), 255, np.uint8)
                stabilization = "reference" if index == 0 and stabilize else "disabled"
                if stabilizer is not None and index > 0:
                    frame, gray, valid, stabilization = stabilizer.apply(frame, gray)
                    if valid is None:
                        failed_stabilization += 1
                aligned = valid is not None
                measurable = index > 0 and aligned and previous_aligned
                sample_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_and(valid, previous_valid)) if measurable else roi_mask.copy()
                magnitudes = np.empty(0, np.float32)
                qualified = 0
                flow = np.zeros((height, width, 2), np.float32)
                annotated = frame.copy()
                if algorithm == "dense" and measurable:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **dense_params)
                    finite = np.isfinite(flow).all(axis=2)
                    flow[~finite] = 0
                    magnitude = np.linalg.norm(flow, axis=2)
                    magnitudes = magnitude[(sample_mask > 0) & finite]
                    flow[sample_mask == 0] = 0
                elif algorithm == "sparse":
                    if not measurable:
                        trajectories = []
                    elif trajectories:
                        p0 = np.float32([track["points"][-1] for track in trajectories]).reshape(-1, 1, 2)
                        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
                        if p1 is not None and st1 is not None:
                            p1_for_back = p1.copy()
                            p1_for_back[~np.isfinite(p1_for_back)] = 0
                            p0r, st0, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1_for_back, None, **lk)
                        else:
                            p0r = st0 = None
                        kept, displacement = [], []
                        if p0r is not None and st0 is not None:
                            good = (st1.ravel() == 1) & (st0.ravel() == 1) & np.isfinite(p1.reshape(-1, 2)).all(axis=1)
                            good &= np.abs(p0 - p0r).reshape(-1, 2).max(axis=1) < fb_threshold
                            for track, new, old, passed in zip(trajectories, p1.reshape(-1, 2), p0.reshape(-1, 2), good):
                                if not passed:
                                    continue
                                x, y = map(float, new)
                                ox, oy = map(float, old)
                                if not (0 <= x < width and 0 <= y < height and 0 <= ox < width and 0 <= oy < height):
                                    continue
                                if sample_mask[int(y), int(x)] == 0 or sample_mask[int(oy), int(ox)] == 0:
                                    continue
                                dx, dy = x - ox, y - oy
                                step_magnitude = math.hypot(dx, dy)
                                displacement.append(step_magnitude)
                                track["points"].append((x, y))
                                track["points"] = track["points"][-trajectory_len:]
                                kept.append(track)
                                start = track["points"][0]
                                net = math.hypot(x - start[0], y - start[1])
                                intervals = len(track["points"]) - 1
                                is_qualified = net > min_displacement
                                qualified += int(is_qualified)
                                if is_qualified:
                                    cv2.polylines(annotated, [np.rint(track["points"]).astype(np.int32)], False,
                                                  (65, 255, 90), track_thickness, cv2.LINE_AA)
                                    endpoint = (round(x), round(y))
                                    radius = max(3, track_thickness + 1)
                                    cv2.circle(annotated, endpoint, radius + 1, (15, 24, 35), -1, cv2.LINE_AA)
                                    cv2.circle(annotated, endpoint, radius, (75, 95, 255), -1, cv2.LINE_AA)
                                track_rows.writerow(dict(frame_index=index, time_sec=index / fps, track_id=track["id"], x_px=x, y_px=y, dx_px=dx, dy_px=dy, speed_px_per_s=step_magnitude * fps, window_intervals=intervals, net_window_speed_px_per_s=net * fps / intervals, qualified=int(is_qualified)))
                        trajectories = kept
                        magnitudes = np.asarray(displacement, np.float32)
                    if aligned and (index % detect_interval == 0 or not trajectories):
                        feature_mask = cv2.bitwise_and(roi_mask, valid)
                        for track in trajectories:
                            x, y = track["points"][-1]
                            cv2.circle(feature_mask, (round(x), round(y)), reseed_radius, 0, -1)
                        remaining = max_corners - len(trajectories)
                        points = cv2.goodFeaturesToTrack(gray, mask=feature_mask, maxCorners=remaining, qualityLevel=quality, minDistance=min_distance, blockSize=block_size) if remaining > 0 else None
                        if points is not None:
                            for x, y in points.reshape(-1, 2):
                                trajectories.append({"id": next_id, "points": [(float(x), float(y))]})
                                next_id += 1
                moving = magnitudes[magnitudes > threshold]
                mean = float(moving.mean()) if moving.size else 0.0
                maximum = float(moving.max()) if moving.size else 0.0
                energy = float(np.square(magnitudes.astype(np.float64)).sum())
                measurement_valid = measurable and magnitudes.size > 0
                status = "reference" if index == 0 else "ok" if measurement_valid else "alignment_unavailable" if not measurable else "no_valid_samples"
                row = dict(frame_index=index, time_sec=index / fps, measurement_valid=int(measurement_valid), status=status, stabilization=stabilization,
                           sample_count=int(magnitudes.size), moving_count=int(moving.size), moving_fraction=float(moving.size / magnitudes.size) if magnitudes.size else 0,
                           mean_displacement_px_per_frame=mean, max_displacement_px_per_frame=maximum, mean_speed_px_per_s=mean * fps, max_speed_px_per_s=maximum * fps,
                           flow_energy_px2_per_frame2=energy, qualified_tracks=qualified)
                rows.writerow(row)
                if measurement_valid:
                    measured_frames += 1
                    total_mean_speed += mean * fps
                    max_speed = max(max_speed, maximum * fps)
                    total_energy += energy
                if algorithm == "dense":
                    visual = _dense_panels(frame, gray, flow, bounds, threshold, **dense_visual)
                else:
                    features = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    for track in trajectories:
                        x, y = track["points"][-1]
                        cv2.circle(features, (round(x), round(y)), 2, (90, 225, 95), -1)
                    x0, y0, x1, y1 = bounds
                    cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (45, 210, 230), 2)
                    # Keep the first panel clean and in track coordinates so that
                    # any frame can serve as the background of a later heatmap.
                    visual = np.vstack((np.hstack((frame.copy(), _panel(gray, "Grayscale"))),
                                        np.hstack((_panel(features, f"Corners: {len(trajectories)}"), _panel(annotated, f"Tracks: {qualified} / {mean * fps:.2f} px/s")))))
                if status not in {"ok", "reference"}:
                    cv2.putText(visual, f"NO MEASUREMENT: {status}", (8, visual.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 80, 255), 1, cv2.LINE_AA)
                writer.send(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
                if index == 0 or (measurement_valid and not preview_written):
                    encoded_ok, encoded = cv2.imencode(".jpg", visual)
                    if encoded_ok:
                        encoded.tofile(str(out_dir / preview_file))
                        preview_written = measurement_valid
                prev_gray = gray.copy()
                previous_valid = valid if valid is not None else np.zeros((height, width), np.uint8)
                previous_aligned = aligned
                index += 1
                if index % 10 == 0:
                    progress(min(95, 5 + (90 * index / total if total else 90 * index / (index + 100))), f"已处理 {index}" + (f" / {total}" if total else "") + " 帧")
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[:2] != (original_height, original_width):
                    raise ValueError("视频帧尺寸发生变化，无法保持同一坐标系；请先转码为固定尺寸")
                if (width, height) != (original_width, original_height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.close()
        writer = None
        if (out_dir / video_file).stat().st_size == 0:
            raise RuntimeError("视频编码失败，没有生成有效 MP4")
        if total and index < total:
            warnings.append(f"声明 {total} 帧，实际解码 {index} 帧；源文件可能不完整或末尾解码失败。")
        if failed_stabilization:
            warnings.append(f"{failed_stabilization} 帧防抖失败；失败帧及跨失败边界的光流测量已排除。")
        if not measured_frames:
            warnings.append("没有可用的帧间光流测量；请检查视频长度、纹理、ROI 和防抖设置。")
        notes = ["像素单位以处理分辨率为准；向右为 +x、向下为 +y。未经尺度标定，不能解释为 m/s。",
                 "均值及最大值仅统计超过噪声阈值的样本；没有移动样本时为 0；measurement_valid=0 的行不参与汇总。",
                 "光流能量为 ROI 内有效向量位移平方和（px²/frame²），不代表物理动能；稀疏与稠密能量不能直接比较。",
                 "导出视频保留输入帧率与首帧，移除音轨；可变帧率输入按标称固定帧率分析。",
                 "四宫格输出为处理宽高的 2 倍；默认原尺寸处理时即为输入宽高的 2 倍，并非将单幅图放大。",
                 "轨迹线宽、稠密向量绘制倍数与幅值显示增益仅改变可视化，不改变位移、速度或光流能量。"]
        effective_params = dict(algorithm=algorithm, stabilize=stabilize, max_width=max_width,
                                noise_threshold=threshold, fps_override=params.get("fps_override"), fps=fps,
                                roi=dict(x=bounds[0] / width, y=bounds[1] / height,
                                         width=(bounds[2] - bounds[0]) / width, height=(bounds[3] - bounds[1]) / height))
        if algorithm == "sparse":
            effective_params.update(max_corners=max_corners, quality_level=quality, min_distance=min_distance,
                                    block_size=block_size, reseed_radius=reseed_radius, track_thickness=track_thickness,
                                    trajectory_len=trajectory_len, detect_interval=detect_interval,
                                    min_track_displacement=min_displacement, fb_threshold=fb_threshold,
                                    lk_win_size=15, lk_max_level=2, lk_iterations=10, lk_epsilon=0.03)
            notes.append("max_corners 是同时保留的轨迹上限；桌面脚本的 maxCorners 是每次重检测新增上限，二者在长视频中不同。")
            notes.append("轨迹图仅绘制窗口首尾净位移超过 min_track_displacement 的轨迹；统计仍使用有效帧间位移与原有噪声阈值。")
        else:
            effective_params.update({f"dense_{key}": value for key, value in dense_params.items()})
            effective_params.update({f"dense_{key}": value for key, value in dense_visual.items()})
            notes.append("稠密光流参数不会随防抖开关自动改变；方向色相不增强，幅值显示增强后截断至 255，避免颜色溢出。")
        metrics = dict(algorithm=algorithm, frames=index, measured_frames=measured_frames, fps=fps, duration_seconds=index / fps,
                       measurement_span_seconds=max(0, index - 1) / fps, original_width=original_width, original_height=original_height,
                       width=width, height=height, output_width=width * 2, output_height=height * 2,
                       processing_mode="native" if (width, height) == (original_width, original_height) else "downscaled",
                       parameters=effective_params, scale_x=width / original_width, scale_y=height / original_height,
                       mean_speed_px_per_s=total_mean_speed / measured_frames if measured_frames else 0.0,
                       max_speed_px_per_s=max_speed, total_flow_energy_px2_per_frame2=total_energy,
                       stabilization_failed_frames=failed_stabilization, roi_processed_pixels=list(bounds), noise_threshold_px_per_frame=threshold)
        files = [dict(name=video_file, label="处理视频 MP4", kind="video"), dict(name=csv_file, label="逐帧数据 CSV", kind="data"),
                 dict(name=summary_file, label="分析摘要 JSON", kind="data"), dict(name=preview_file, label="分析预览", kind="image")]
        result = dict(algorithm=algorithm, files=files, metrics=metrics, warnings=warnings, notes=notes, source=SOURCE)
        if algorithm == "sparse":
            result["files"].append(dict(name=tracks_file, label="特征轨迹 CSV", kind="data"))
        else:
            (out_dir / tracks_file).unlink(missing_ok=True)
        (out_dir / summary_file).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        progress(100, "视频、逐帧数据与摘要已生成")
        return result
    finally:
        cap.release()
        if writer is not None:
            writer.close()
