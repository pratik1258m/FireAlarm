"""
Core computer vision fire detector combining multi-color-space rules,
MOG2 motion gating, frame-diff flicker, and temporal tracking.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.config import FireDetectionConfig
from src.utils import _clamp


@dataclass
class _Track:
    """Per-blob temporal history used to measure real fire flicker."""

    track_id: int
    cx: float
    cy: float
    area: int
    bbox: Tuple[int, int, int, int]
    value: float = 0.0
    score: float = 0.0
    misses: int = 0
    frames: int = 1

    def __post_init__(self) -> None:
        self.area_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.iou_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.value_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.cy_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.area_hist.append(float(self.area))
        self.value_hist.append(float(self.value))
        self.cy_hist.append(float(self.cy))


class FireDetector:
    """
    Research-backed fire detector combining:
    - Chen et al. (2004) RGB channel ordering (R >= G > B)
    - Celik & Demirel (2009) adaptive YCbCr fire chrominance model
    - MOG2 background subtraction + per-blob edge frame-diff flicker
    - Per-blob temporal flicker (area + IoU + luminance oscillation)
    """

    HISTORY = 16
    WEIGHTS = {"core": 0.26, "chroma": 0.22, "temporal": 0.26, "variation": 0.08, "flow": 0.18}

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._prev_mask: Optional[np.ndarray] = None
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id = 0
        self._fire_first_seen: Optional[float] = None
        self._fire_last_seen: float = 0.0
        self._alarm_active = False
        self._last_score = 0.0
        self._last_flicker = 0.0
        self._last_t = time.time()
        self._fps_times: List[float] = []
        self.debug_view = config.show_masks
        self.debug_image: Optional[np.ndarray] = None
        self.last_candidates: List[dict] = []

        self._bg_sub: Optional[cv2.BackgroundSubtractor] = None
        if config.use_bg_subtraction:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=config.bg_history,
                varThreshold=config.bg_var_threshold,
                detectShadows=False,
            )
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_scene_mean_y: Optional[float] = None
        self._light_jump_frames: int = 0

    # --------------------------------------------------------- segmentation
    def _segment(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (mask_fire, mask_hot, mask_halo, ycrcb)."""
        cfg = self._cfg
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
        b_ch, g_ch, r_ch = cv2.split(bgr)

        # Scene-adaptive means (Celik & Demirel, 2009)
        y_mean = float(y_ch.mean())
        cr_mean = float(cr_ch.mean())
        cb_mean = float(cb_ch.mean())

        # Global lighting jump detection (room light switch or car headlights)
        if cfg.global_light_filter and self._prev_scene_mean_y is not None:
            if abs(y_mean - self._prev_scene_mean_y) > cfg.max_scene_luma_jump:
                self._light_jump_frames = 3
        self._prev_scene_mean_y = y_mean

        y_f = y_ch.astype(np.int16)
        cr_f = cr_ch.astype(np.int16)
        cb_f = cb_ch.astype(np.int16)
        r_f = r_ch.astype(np.int16)
        g_f = g_ch.astype(np.int16)
        b_f = b_ch.astype(np.int16)

        # 1. Chen et al. RGB rule: R >= G > B with G - B gap (rejects red cloth)
        rgb_fire = (r_f >= cfg.r_min) & (r_f >= g_f) & ((g_f - b_f) >= cfg.gb_diff_min)

        # 2. Celik & Demirel YCbCr rule: brighter than room + warm chrominance
        ycrcb_fire = (
            (y_f > cb_f)
            & (cr_f > cb_f)
            & (y_f > y_mean)
            & (cr_f > cr_mean)
            & (cb_f < cb_mean)
            & ((cr_f - cb_f) >= cfg.ycbcr_tau)
        )

        mask_fire = (rgb_fire & ycrcb_fire).astype(np.uint8) * 255
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, self._kernel_open, iterations=1)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, self._kernel_close, iterations=1)

        # Relaxed fire glow (halo) used to validate small overexposed match cores
        halo = (
            (r_f >= max(160, cfg.r_min - 30))
            & (r_f >= g_f)
            & ((g_f - b_f) >= max(12, cfg.gb_diff_min - 8))
            & ((cr_f - cb_f) >= max(22, cfg.ycbcr_tau - 13))
            & (y_f > y_mean)
        )
        mask_halo = halo.astype(np.uint8) * 255

        # White-hot core: very high luminance + warm tint (R >= B)
        hot = (y_f >= cfg.hot_y_min) & (r_f >= g_f - 10) & (r_f >= b_f)
        mask_hot = hot.astype(np.uint8) * 255

        return mask_fire, mask_hot, mask_halo, ycrcb

    @staticmethod
    def _compact_hot_mask(mask_hot: np.ndarray, max_px: int) -> np.ndarray:
        """Keep only compact pale blobs (rejects bright walls/windows)."""
        out = np.zeros_like(mask_hot)
        contours, _ = cv2.findContours(mask_hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if cv2.countNonZero(mask_hot[y:y + bh, x:x + bw]) <= max_px:
                cv2.drawContours(out, [cnt], -1, 255, -1)
        return out

    # ------------------------------------------------------------- candidates
    def _build_candidate(
        self,
        cnt: np.ndarray,
        ycrcb: np.ndarray,
        src_mask: np.ndarray,
        mask_halo: np.ndarray,
        combined: np.ndarray,
        lo_area: int,
        hi_area: int,
        is_hot: bool,
    ) -> Optional[dict]:
        cfg = self._cfg
        h, w = src_mask.shape[:2]
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 3 or bh < 3:
            return None
        roi_mask = src_mask[y:y + bh, x:x + bw]
        area = int(cv2.countNonZero(roi_mask))
        if area <= 0 or area < lo_area or area > hi_area:
            return None
        if bw / (bh + 1e-6) > cfg.max_aspect:
            return None

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = cv2.contourArea(cnt) / (hull_area + 1e-6)
        if solidity < cfg.min_solidity or solidity > cfg.max_solidity:
            return None

        # Measure fire-colored halo around the blob (padded so dilation isn't clipped)
        reach = max(7, int(0.55 * max(bw, bh)))
        reach += 1 if reach % 2 == 0 else 0
        r_pad = reach // 2
        ry0, ry1 = max(0, y - r_pad), min(h, y + bh + r_pad)
        rx0, rx1 = max(0, x - r_pad), min(w, x + bw + r_pad)
        padded_mask = src_mask[ry0:ry1, rx0:rx1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
        ring = cv2.subtract(cv2.dilate(padded_mask, kernel), padded_mask)
        ring_px = cv2.countNonZero(ring)
        glow = 0.0
        if ring_px > 10:
            core_ring = cv2.bitwise_and(ring, mask_halo[ry0:ry1, rx0:rx1])
            glow = cv2.countNonZero(core_ring) / float(ring_px)

        if is_hot:
            if glow < cfg.min_glow_ratio:
                return None
            # Radial Thermal Gradient check:
            # Real flame cools outward (white core -> warm orange/red halo).
            # Bare white LED lamps and specular highlights lack warm chrominance in their surrounding halo.
            if ring_px > 12:
                cr_pad = ycrcb[ry0:ry1, rx0:rx1, 1].astype(np.int16)
                cb_pad = ycrcb[ry0:ry1, rx0:rx1, 2].astype(np.int16)
                ring_diff = float((cr_pad[ring > 0] - cb_pad[ring > 0]).mean())
                if ring_diff < float(cfg.ycbcr_tau) - 18.0 and glow < 0.16:
                    return None

        if is_hot:
            pad = max(4, int(0.35 * max(bw, bh)))
            y0, y1 = max(0, y - pad), min(h, y + bh + pad)
            x0, x1 = max(0, x - pad), min(w, x + bw + pad)
            ycrcb_roi = ycrcb[y0:y1, x0:x1]
            halo_roi = mask_halo[y0:y1, x0:x1]
            blob_roi = combined[y0:y1, x0:x1]
        else:
            ycrcb_roi = ycrcb[y:y + bh, x:x + bw]
            halo_roi = mask_halo[y:y + bh, x:x + bw]
            blob_roi = roi_mask

        y_vals = ycrcb_roi[:, :, 0]
        glow_px = y_vals[halo_roi > 0]
        if glow_px.size < 10:
            glow_px = y_vals[blob_roi > 0]
        mean_y = float(glow_px.mean()) if glow_px.size else 0.0

        moments = cv2.moments(cnt)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + bw / 2.0, y + bh / 2.0

        return {
            "bbox": (x, y, bw, bh),
            "source": "hot" if is_hot else "fire",
            "area": area,
            "centroid": (cx, cy),
            "roi_mask": roi_mask,
            "mean_v": mean_y,
            "s_core": self._score_core(ycrcb_roi, blob_roi, glow),
            "s_chroma": self._score_chroma(ycrcb_roi, blob_roi, halo_roi),
            "s_variation": self._score_variation(ycrcb_roi, blob_roi, halo_roi),
            "s_flow": 0.35,
            "score": 0.0,
        }

    @staticmethod
    def _contained(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        x0, y0 = max(ix, ox), max(iy, oy)
        x1, y1 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
        if x1 <= x0 or y1 <= y0:
            return False
        return ((x1 - x0) * (y1 - y0)) / float(max(1, iw * ih)) > 0.85

    # ------------------------------------------------------------- sub-scores
    def _score_core(self, ycrcb_roi: np.ndarray, blob_roi: np.ndarray, glow: float) -> float:
        """Luminance peak check: real flames are bright light sources."""
        y_vals = ycrcb_roi[:, :, 0][blob_roi > 0]
        if y_vals.size < 8:
            return 0.0
        p95 = float(np.percentile(y_vals, 95))
        if p95 < 200.0:
            return 0.0
        score = _clamp((p95 - 195.0) / 50.0)
        hot_frac = float((y_vals >= self._cfg.hot_y_min).mean())
        if hot_frac > 0.90 and glow < 0.12:
            score *= 0.30  # uniformly blown-out white object with no fire halo
        return _clamp(score)

    def _score_chroma(self, ycrcb_roi: np.ndarray, blob_roi: np.ndarray, halo_roi: np.ndarray) -> float:
        """Celik Cr - Cb fire chrominance strength."""
        sel = (blob_roi > 0) | (halo_roi > 0)
        if int(sel.sum()) < 8:
            return 0.0
        cr = ycrcb_roi[:, :, 1][sel].astype(np.float32)
        cb = ycrcb_roi[:, :, 2][sel].astype(np.float32)
        diff = cr - cb
        p80 = float(np.percentile(diff, 80))
        return _clamp((p80 - 20.0) / 55.0)

    def _score_variation(self, ycrcb_roi: np.ndarray, blob_roi: np.ndarray, halo_roi: np.ndarray) -> float:
        """Flames have internal luminance and chrominance gradients; flat objects don't."""
        sel = (blob_roi > 0) | (halo_roi > 0)
        if int(sel.sum()) < 8:
            return 0.0
        y_v = ycrcb_roi[:, :, 0][sel].astype(np.float32)
        cr_v = ycrcb_roi[:, :, 1][sel].astype(np.float32)
        return _clamp(0.5 * (float(y_v.std()) / 30.0) + 0.5 * (float(cr_v.std()) / 15.0))

    def _score_temporal(self, track: _Track) -> float:
        """Fire flickers in shape/area and/or luminance.

        Uses a weighted sum instead of a product so overexposed white-hot
        flames (whose core clips at 255) still score strongly on shape flicker.
        """
        if track.frames < self._cfg.warmup_frames:
            return 0.0
        areas = np.array(track.area_hist, dtype=np.float32)
        mean_area = float(areas.mean())
        if mean_area <= 1.0:
            return 0.0
        area_score = _clamp(float(areas.std() / mean_area) / 0.08)
        if track.iou_hist:
            shape_score = _clamp((1.0 - float(np.mean(track.iou_hist))) / 0.22)
        else:
            shape_score = 0.0
        shape_change = 0.55 * area_score + 0.45 * shape_score

        values = np.array(track.value_hist, dtype=np.float32)
        mean_value = float(values.mean())
        brightness_change = 0.0
        if mean_value > 1.0:
            brightness_change = _clamp(float(values.std() / mean_value) / 0.025)

        return _clamp(0.70 * shape_change + 0.30 * brightness_change)

    @staticmethod
    def _score_frame_diff(
        gray_curr: Optional[np.ndarray],
        gray_prev: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """Per-blob edge frame-differencing flicker score.

        Pads the bbox by 2px so it captures the flickering flame boundary
        without pulling in unrelated moving objects nearby.
        """
        if gray_curr is None or gray_prev is None:
            return 0.35
        x, y, bw, bh = bbox
        h, w = gray_curr.shape
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        if y1 <= y0 or x1 <= x0:
            return 0.35
        curr_roi = gray_curr[y0:y1, x0:x1].astype(np.float32)
        prev_roi = gray_prev[y0:y1, x0:x1].astype(np.float32)
        diff = np.abs(curr_roi - prev_roi)
        edge_diff = float(np.percentile(diff, 85))
        return _clamp(edge_diff / 22.0)

    # ------------------------------------------------------------- tracking
    def _match_tracks(self, candidates: List[dict], now: float) -> List[dict]:
        for track in self._tracks.values():
            track.misses += 1

        matched_ids: Set[int] = set()
        for cand in candidates:
            cx, cy = cand["centroid"]
            best: Optional[_Track] = None
            best_dist = float("inf")
            for track in self._tracks.values():
                if track.track_id in matched_ids:
                    continue
                smaller = max(1, min(track.area, cand["area"]))
                larger = max(track.area, cand["area"])
                if larger / float(smaller) > 8.0:
                    continue
                dist = math.hypot(track.cx - cx, track.cy - cy)
                reach = max(35.0, 0.75 * math.sqrt(max(track.area, 1)))
                if dist < reach and dist < best_dist:
                    best, best_dist = track, dist

            x, y, bw, bh = cand["bbox"]
            if best is None:
                track = _Track(
                    self._next_track_id, cx, cy, cand["area"], cand["bbox"],
                    value=cand.get("mean_v", 0.0),
                )
                self._next_track_id += 1
                self._tracks[track.track_id] = track
                matched_ids.add(track.track_id)
                cand["track"] = track
            else:
                matched_ids.add(best.track_id)
                prev_crop = None
                if self._prev_mask is not None:
                    ph, pw = self._prev_mask.shape[:2]
                    y0, y1 = max(0, y), min(ph, y + bh)
                    x0, x1 = max(0, x), min(pw, x + bw)
                    if y1 > y0 and x1 > x0:
                        prev_crop = self._prev_mask[y0:y1, x0:x1]

                best.misses = 0
                best.frames += 1
                best.cx, best.cy = cx, cy
                best.area = cand["area"]
                best.bbox = cand["bbox"]
                best.area_hist.append(float(cand["area"]))
                best.value_hist.append(float(cand.get("mean_v", 0.0)))
                best.cy_hist.append(float(cy))
                if prev_crop is not None and prev_crop.shape == cand["roi_mask"].shape:
                    inter = cv2.countNonZero(cv2.bitwise_and(cand["roi_mask"], prev_crop))
                    union = cv2.countNonZero(cv2.bitwise_or(cand["roi_mask"], prev_crop))
                    if union > 0:
                        best.iou_hist.append(inter / float(union))
                cand["track"] = best

        dead = [tid for tid, t in self._tracks.items() if t.misses > 12]
        for tid in dead:
            del self._tracks[tid]
        return candidates

    def _update_confidence(self, candidates: List[dict]) -> Tuple[float, float]:
        for track in self._tracks.values():
            if track.misses > 0:
                track.score *= 0.72

        best = 0.0
        best_flicker = 0.0
        for cand in candidates:
            track: _Track = cand["track"]
            flicker = self._score_temporal(track)
            flow = cand.get("s_flow", 0.35)
            effective_flicker = max(flicker, 0.75 * flow)
            if cand["s_core"] < 0.25 or cand["s_chroma"] < 0.25:
                instant = 0.0
            else:
                instant = (
                    self.WEIGHTS["core"] * cand["s_core"]
                    + self.WEIGHTS["chroma"] * cand["s_chroma"]
                    + self.WEIGHTS["temporal"] * flicker
                    + self.WEIGHTS["variation"] * cand["s_variation"]
                    + self.WEIGHTS["flow"] * flow
                )
            if track.frames <= 1:
                track.score = instant
            else:
                alpha = self._cfg.score_smoothing
                track.score = (1.0 - alpha) * track.score + alpha * instant
            cand["score"] = track.score
            if track.score > best:
                best = track.score
                best_flicker = effective_flicker
        return best, best_flicker

    # ------------------------------------------------------------- main entry
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        now = time.time()
        h_orig, w_orig = frame.shape[:2]
        target_w = self._cfg.process_width
        scale = target_w / float(w_orig) if w_orig > target_w else 1.0
        proc = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale))) if scale < 1.0 else frame
        blurred = cv2.GaussianBlur(proc, (3, 3), 0)

        # --- Background subtraction: freeze learning while alarm is active ---
        if self._bg_sub is not None:
            lr = 0.0 if self._alarm_active else 0.002
            fg_mask = self._bg_sub.apply(blurred, learningRate=lr)
            fg_dilated = cv2.dilate(fg_mask, self._kernel_close, iterations=2)
        else:
            fg_dilated = None

        gray_curr = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray_prev = self._prev_gray
        self._prev_gray = gray_curr

        mask_fire, raw_hot, mask_halo, ycrcb = self._segment(blurred)

        h, w = mask_fire.shape[:2]
        frame_area = float(h * w)
        min_px = max(self._cfg.min_contour_area, int(self._cfg.min_area_ratio * frame_area))
        max_px = int(self._cfg.max_area_ratio * frame_area)
        max_hot_px = int(self._cfg.max_hot_area_ratio * frame_area)

        mask_hot = self._compact_hot_mask(raw_hot, max_hot_px)
        combined = cv2.bitwise_or(mask_fire, mask_hot)

        candidates: List[dict] = []
        for src_mask, lo_area, hi_area, is_hot in (
            (mask_fire, min_px, max_px, False),
            (mask_hot, self._cfg.min_hot_area, max_hot_px, True),
        ):
            contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cand = self._build_candidate(
                    cnt, ycrcb, src_mask, mask_halo, combined, lo_area, hi_area, is_hot
                )
                if cand is not None:
                    candidates.append(cand)

        # Suppress transient candidate spikes during sudden whole-room lighting changes
        if self._light_jump_frames > 0:
            self._light_jump_frames -= 1
            candidates = []

        # --- Motion gate: drop blobs that aren't in a moving region (MOG2) ---
        if fg_dilated is not None and self._cfg.motion_min_overlap > 0:
            gated = []
            for cand in candidates:
                x, y, bw, bh = cand["bbox"]
                fg_roi = fg_dilated[y : y + bh, x : x + bw]
                total = bw * bh
                if total > 0 and cv2.countNonZero(fg_roi) / float(total) >= self._cfg.motion_min_overlap:
                    gated.append(cand)
            candidates = gated

        self.last_candidates = candidates

        # --- Per-candidate edge frame-diff flicker score ---
        if self._cfg.use_optical_flow:
            for cand in candidates:
                cand["s_flow"] = self._score_frame_diff(gray_curr, gray_prev, cand["bbox"])

        candidates = self._match_tracks(candidates, now)
        confidence, flicker = self._update_confidence(candidates)
        self._last_score = confidence
        self._last_flicker = flicker
        self._prev_mask = combined.copy()

        fire_confirmed = confidence >= self._cfg.min_fire_score and flicker >= self._cfg.min_flicker
        alarm = self._update_persistence(fire_confirmed, now)

        annotated = frame.copy()
        drawn: List[Tuple[int, int, int, int]] = []
        inv = 1.0 / scale
        for cand in sorted(candidates, key=lambda c: -c.get("score", 0.0)):
            score = cand.get("score", 0.0)
            if score < self._cfg.clear_score or cand["track"].frames < self._cfg.warmup_frames:
                continue
            x, y, bw, bh = cand["bbox"]
            if scale < 1.0:
                x, y, bw, bh = int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)
            if any(self._contained((x, y, bw, bh), box) for box in drawn):
                continue
            drawn.append((x, y, bw, bh))
            fired = score >= self._cfg.min_fire_score
            colour = (0, 70, 255) if fired else (0, 190, 255)

            # High-tech targeting bracket
            self._draw_target_box(annotated, x, y, bw, bh, colour, fired, score)

            if fired:
                roi = annotated[y:y + bh, x:x + bw]
                tint = roi.copy()
                tint[:] = (0, 0, 200)
                annotated[y:y + bh, x:x + bw] = cv2.addWeighted(roi, 0.80, tint, 0.20, 0)

        annotated = self._draw_hud(annotated, alarm, confidence, len(drawn) > 0)
        if self.debug_view:
            dbg = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            for cand in candidates:
                x, y, bw, bh = cand["bbox"]
                cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
                cv2.putText(
                    dbg, f"{cand.get('score', 0.0) * 100:.0f}%", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
                )
            self.debug_image = dbg
        else:
            self.debug_image = None
        return annotated, alarm

    # ------------------------------------------------------------ persistence
    def _update_persistence(self, fire_detected: bool, now: float) -> bool:
        if fire_detected:
            self._fire_last_seen = now
            if self._fire_first_seen is None:
                self._fire_first_seen = now
            if (now - self._fire_first_seen) >= self._cfg.persistence_threshold:
                self._alarm_active = True
        else:
            if (now - self._fire_last_seen) > self._cfg.persistence_grace:
                self._fire_first_seen = None
            cleared = (
                self._last_score < self._cfg.clear_score
                or self._last_flicker < self._cfg.min_flicker * 0.5
            )
            if cleared and (now - self._fire_last_seen) > self._cfg.clear_delay:
                self._alarm_active = False
        return self._alarm_active

    @staticmethod
    def _draw_target_box(
        frame: np.ndarray,
        x: int,
        y: int,
        bw: int,
        bh: int,
        colour: Tuple[int, int, int],
        fired: bool,
        score: float,
    ) -> None:
        """Draws a high-tech corner-bracket targeting reticle with solid pill label."""
        # 1. Subtle border
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 1, cv2.LINE_AA)

        # 2. Bold L-shaped targeting corners
        t_len = max(8, min(bw, bh) // 4)
        thickness = 3 if fired else 2
        # Top-Left
        cv2.line(frame, (x, y), (x + t_len, y), colour, thickness)
        cv2.line(frame, (x, y), (x, y + t_len), colour, thickness)
        # Top-Right
        cv2.line(frame, (x + bw, y), (x + bw - t_len, y), colour, thickness)
        cv2.line(frame, (x + bw, y), (x + bw, y + t_len), colour, thickness)
        # Bottom-Left
        cv2.line(frame, (x, y + bh), (x + t_len, y + bh), colour, thickness)
        cv2.line(frame, (x, y + bh), (x, y + bh - t_len), colour, thickness)
        # Bottom-Right
        cv2.line(frame, (x + bw, y + bh), (x + bw - t_len, y + bh), colour, thickness)
        cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - t_len), colour, thickness)

        # 3. Solid high-contrast tag pill
        label = f"{'FIRE' if fired else 'SUSPECT'} {score * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        pill_y2 = max(0, y - 4)
        pill_y1 = max(0, pill_y2 - th - 10)
        pill_x2 = min(frame.shape[1], x + tw + 14)
        # Dark backing plate
        cv2.rectangle(frame, (x, pill_y1), (pill_x2, pill_y2), (18, 18, 22), -1)
        cv2.rectangle(frame, (x, pill_y1), (pill_x2, pill_y2), colour, 1, cv2.LINE_AA)
        cv2.putText(
            frame, label, (x + 7, pill_y2 - 5),
            cv2.FONT_HERSHEY_DUPLEX, 0.45, colour, 1, cv2.LINE_AA
        )

    # -------------------------------------------------------------------- HUD
    def _draw_hud(
        self,
        frame: np.ndarray,
        alarm: bool,
        confidence: float,
        has_suspects: bool = False,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Top HUD Bar (Frosted glass header)
        top_h = 42
        top_bar = frame[0:top_h, 0:w].copy()
        top_plate = np.full_like(top_bar, (16, 18, 24))
        frame[0:top_h, 0:w] = cv2.addWeighted(top_plate, 0.82, top_bar, 0.18, 0)
        cv2.line(frame, (0, top_h), (w, top_h), (45, 50, 62), 1)

        # Top Left: Status Dot & System Title
        now_t = time.time()
        pulse = (int(now_t * 3) % 2 == 0)
        if alarm:
            dot_color = (0, 0, 240) if pulse else (0, 60, 200)
            tag_text = "FIRE GUARD AI  |  ZONE-01"
        elif has_suspects:
            dot_color = (0, 190, 255)
            tag_text = "FIRE GUARD AI  |  ANALYZING"
        else:
            dot_color = (40, 215, 60)
            tag_text = "FIRE GUARD AI  |  ZONE-01"

        cv2.circle(frame, (20, 21), 6, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (20, 21), 8, dot_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame, tag_text, (36, 26),
            cv2.FONT_HERSHEY_DUPLEX, 0.52, (240, 242, 245), 1, cv2.LINE_AA
        )

        # Top Center: Timestamp
        dt_str = time.strftime("%Y-%m-%d  %H:%M:%S")
        (dt_w, _), _ = cv2.getTextSize(dt_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            frame, dt_str, (max(w // 2 - dt_w // 2, 220), 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 195), 1, cv2.LINE_AA
        )

        # Top Right: Metrics (FPS & Confidence Meter)
        fps_val = 0.0
        if self._cfg.show_fps:
            dt = now_t - self._last_t
            self._last_t = now_t
            if dt > 0:
                self._fps_times.append(1.0 / dt)
            if len(self._fps_times) > 30:
                self._fps_times.pop(0)
            fps_val = sum(self._fps_times) / len(self._fps_times) if self._fps_times else 0.0

        fps_str = f"FPS {fps_val:4.1f}"
        (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(
            frame, fps_str, (w - fps_w - 18, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 205, 215), 1, cv2.LINE_AA
        )

        # Confidence Bar & Text
        bar_w = 90
        bar_x = w - fps_w - bar_w - 45
        bar_y1 = 16
        bar_y2 = 26
        fill_w = int(bar_w * _clamp(confidence))
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + bar_w, bar_y2), (35, 38, 48), -1)
        if confidence > 0.60:
            bar_col = (0, 70, 255)
        elif confidence > 0.35:
            bar_col = (0, 190, 255)
        else:
            bar_col = (50, 200, 50)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + fill_w, bar_y2), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + bar_w, bar_y2), (80, 85, 100), 1)

        conf_str = f"{confidence * 100:3.0f}%"
        cv2.putText(
            frame, conf_str, (bar_x - 48, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, bar_col, 1, cv2.LINE_AA
        )

        # 2. Bottom HUD Bar
        bot_h = 44
        bot_bar = frame[h - bot_h:h, 0:w].copy()
        bot_plate = np.full_like(bot_bar, (14, 16, 20))
        frame[h - bot_h:h, 0:w] = cv2.addWeighted(bot_plate, 0.85, bot_bar, 0.15, 0)
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (45, 50, 62), 1)

        # Bottom Left: Security State Pill
        if alarm:
            state_bg = (0, 0, 180) if pulse else (0, 0, 220)
            state_text = "*** CRITICAL FIRE DETECTED — EVACUATE! ***"
            state_text_col = (255, 255, 255)
        elif has_suspects:
            state_bg = (0, 120, 180)
            state_text = "ANALYZING POTENTIAL HEAT / FLAME SIGNATURE"
            state_text_col = (255, 240, 210)
        else:
            state_bg = (20, 60, 25)
            state_text = "ONLINE — CONTINUOUS SECTOR MONITORING"
            state_text_col = (100, 240, 120)

        (sw, _), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        pill_rx = min(w - 240, 24 + sw + 20)
        cv2.rectangle(frame, (16, h - bot_h + 8), (pill_rx, h - 8), state_bg, -1)
        cv2.rectangle(frame, (16, h - bot_h + 8), (pill_rx, h - 8), (100, 110, 130), 1, cv2.LINE_AA)
        cv2.putText(
            frame, state_text, (26, h - 16),
            cv2.FONT_HERSHEY_DUPLEX, 0.52, state_text_col, 1, cv2.LINE_AA
        )

        # Bottom Right: Branding & Camera Info
        brand = "CAM ACTIVE  |  Pratik Mishra  v5.5"
        (bw_t, _), _ = cv2.getTextSize(brand, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(
            frame, brand, (w - bw_t - 16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 160, 175), 1, cv2.LINE_AA
        )

        # 3. Full-Frame Hazard Flash Border on Alarm
        if alarm and pulse:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 240), 8)

        return frame
