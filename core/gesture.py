"""
core/gesture.py — Local-first double-clap / double-snap gesture detection.

Like core/wake_word.py, this runs entirely offline on raw microphone PCM —
no audio or gesture data ever leaves the device. It is pure signal
processing (short-time energy + a coarse spectral-centroid heuristic), not
a trained/ML classifier: claps and snaps are told apart from each other,
and from other transient noise (keyboard clicks, mouse clicks), by their
typical duration, loudness and frequency-content profile. All thresholds
are configurable (see GestureConfig) because real-world calibration
depends on the microphone, room, and distance — the defaults here are
reasonable starting points, not measured/tuned against real recordings.

Detection pipeline, per audio chunk fed in:
  1. Slice into fixed 20ms sub-frames (independent of the caller's chunk
     size) and track short-time RMS energy per sub-frame.
  2. Maintain an adaptive noise floor (EMA of RMS while idle) so the
     detector adjusts to a quiet room vs. a noisy one instead of using a
     single fixed volume threshold.
  3. An "impulse" opens when energy spikes well above the floor and closes
     when it decays back down (with hysteresis) or exceeds a max duration
     (anything longer than a real clap/snap — e.g. speech — is discarded).
  4. Each closed impulse is classified as CLAP, SNAP, or discarded, using
     its duration, peak amplitude, and spectral centroid (an FFT-based
     "is the energy concentrated in low/broad or high/narrow frequencies"
     measure — claps are louder and more broadband/low, snaps are quieter,
     shorter, and skew higher-frequency).
  5. Classified single events feed a per-kind double-event timer (two
     events of the SAME kind within [MIN_INTERVAL, MAX_INTERVAL] of each
     other = a confirmed double-clap/double-snap), gated by a shared
     cooldown so one gesture can't fire multiple state transitions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE   = 16000
_SUBFRAME_MS  = 5
_SUBFRAME_LEN = int(SAMPLE_RATE * _SUBFRAME_MS / 1000)   # 80 samples — fine enough that a short
                                                          # snap (a few ms) isn't diluted by getting
                                                          # smeared into a subframe mostly full of silence


class GestureKind(Enum):
    CLAP = "clap"
    SNAP = "snap"


@dataclass
class GestureConfig:
    """All tunable thresholds in one place — see module docstring."""

    enabled_claps: bool = True
    enabled_snaps: bool = True

    # Onset thresholds — multiplier over the adaptive noise floor. Claps are
    # louder, so they need a higher bar to avoid matching ordinary noise;
    # snaps are quieter and need a lower one just to be seen at all, with
    # duration/spectral checks below doing the real discrimination work.
    clap_detection_threshold: float = 6.0
    snap_detection_threshold: float = 3.0
    min_absolute_rms: float = 250.0        # int16 RMS floor so near-silence doesn't cause false onsets

    # Impulse shape — duration in ms, spectral centroid in Hz.
    clap_min_duration_ms: float = 15.0
    clap_max_duration_ms: float = 90.0
    clap_max_centroid_hz: float = 3500.0
    snap_min_duration_ms: float = 3.0
    snap_max_duration_ms: float = 45.0
    snap_min_centroid_hz: float = 2000.0
    max_impulse_duration_ms: float = 150.0  # longer than this: not a clap/snap (speech, etc.) — discard

    # Double-event timing, per gesture kind.
    double_clap_min_interval_ms: float = 120.0
    double_clap_max_interval_ms: float = 600.0
    double_snap_min_interval_ms: float = 100.0
    double_snap_max_interval_ms: float = 500.0

    # Cooldowns / debounce.
    single_event_refractory_ms: float = 80.0    # ignore new onsets right after one closes (echo/tail)
    gesture_cooldown_ms: float = 1200.0         # after a CONFIRMED double-event, ignore further doubles


@dataclass
class _PendingImpulse:
    start_t: float
    peak_rms: float = 0.0
    samples: list = field(default_factory=list)


class GestureDetector:
    """
    feed(pcm_bytes) — raw int16 mono PCM @16kHz, any chunk size.
    Returns GestureKind on a CONFIRMED double-event, else None.

    Also invokes `on_event(kind, is_double)` for every single AND double
    detection if a callback was provided, purely for logging — state
    transitions should only ever be driven by the return value / the
    is_double=True callback, never by single-event notifications.
    """

    def __init__(self, config: Optional[GestureConfig] = None,
                 on_event: Optional[Callable[[GestureKind, bool], None]] = None):
        self.cfg = config or GestureConfig()
        self._on_event = on_event

        self._buf = np.empty(0, dtype=np.int16)
        self._noise_floor = 600.0     # seed value; adapts quickly via EMA
        self._impulse: Optional[_PendingImpulse] = None

        self._pending_single: dict[GestureKind, float] = {}   # kind -> timestamp of unpaired event
        # -inf, not 0.0: time.monotonic() is an arbitrary, implementation-defined
        # epoch (often already a large number) — 0.0 is not "the distant past".
        self._last_impulse_end = float("-inf")
        self._last_double_t = float("-inf")

    # ── public API ───────────────────────────────────────────────────────

    def feed(self, pcm_bytes: bytes) -> Optional[GestureKind]:
        chunk = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._buf = np.concatenate([self._buf, chunk]) if self._buf.size else chunk

        result: Optional[GestureKind] = None
        while self._buf.size >= _SUBFRAME_LEN:
            frame, self._buf = self._buf[:_SUBFRAME_LEN], self._buf[_SUBFRAME_LEN:]
            r = self._process_subframe(frame)
            if r is not None:
                result = r   # keep the last one if somehow >1 resolves in one feed() call
        return result

    # ── internals ────────────────────────────────────────────────────────

    def _process_subframe(self, frame: np.ndarray) -> Optional[GestureKind]:
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-9))

        onset_threshold = self._noise_floor * min(
            self.cfg.clap_detection_threshold if self.cfg.enabled_claps else 1e9,
            self.cfg.snap_detection_threshold if self.cfg.enabled_snaps else 1e9,
        )
        sustain_threshold = max(self._noise_floor * 2.0, self.cfg.min_absolute_rms * 0.5)

        if self._impulse is None:
            # Idle: track noise floor, watch for an onset.
            self._noise_floor = 0.98 * self._noise_floor + 0.02 * rms
            in_refractory = (now - self._last_impulse_end) * 1000 < self.cfg.single_event_refractory_ms
            if (not in_refractory and rms >= self.cfg.min_absolute_rms
                    and rms >= onset_threshold):
                self._impulse = _PendingImpulse(start_t=now, peak_rms=rms, samples=[frame])
            return None

        # Inside an impulse.
        self._impulse.peak_rms = max(self._impulse.peak_rms, rms)
        self._impulse.samples.append(frame)
        duration_ms = (now - self._impulse.start_t) * 1000

        still_active = rms >= sustain_threshold
        too_long = duration_ms >= self.cfg.max_impulse_duration_ms
        if still_active and not too_long:
            return None

        # Impulse closed — classify it.
        closed = self._impulse
        self._impulse = None
        self._last_impulse_end = now
        if too_long:
            return None   # sustained sound (speech, etc.) — not a clap/snap

        return self._classify_and_register(closed, duration_ms, now)

    def _classify_and_register(self, imp: _PendingImpulse, duration_ms: float,
                                now: float) -> Optional[GestureKind]:
        centroid_hz = _spectral_centroid(np.concatenate(imp.samples))
        peak = imp.peak_rms
        cfg = self.cfg

        kind: Optional[GestureKind] = None
        if (cfg.enabled_claps and peak >= self._noise_floor * cfg.clap_detection_threshold
                and cfg.clap_min_duration_ms <= duration_ms <= cfg.clap_max_duration_ms
                and centroid_hz <= cfg.clap_max_centroid_hz):
            kind = GestureKind.CLAP
        elif (cfg.enabled_snaps and peak >= self._noise_floor * cfg.snap_detection_threshold
                and cfg.snap_min_duration_ms <= duration_ms <= cfg.snap_max_duration_ms
                and centroid_hz >= cfg.snap_min_centroid_hz):
            kind = GestureKind.SNAP

        if kind is None:
            return None   # ambiguous impulse (e.g. a keyboard/mouse click) — discard, don't guess

        return self._register_event(kind, now)

    def _register_event(self, kind: GestureKind, now: float) -> Optional[GestureKind]:
        cfg = self.cfg
        min_ms, max_ms = (
            (cfg.double_clap_min_interval_ms, cfg.double_clap_max_interval_ms)
            if kind is GestureKind.CLAP
            else (cfg.double_snap_min_interval_ms, cfg.double_snap_max_interval_ms)
        )

        pending_t = self._pending_single.get(kind)
        if pending_t is not None:
            gap_ms = (now - pending_t) * 1000
            if min_ms <= gap_ms <= max_ms:
                self._pending_single.pop(kind, None)
                if (now - self._last_double_t) * 1000 < cfg.gesture_cooldown_ms:
                    self._log(f"Double {kind.value} ignored — cooldown active")
                    if self._on_event:
                        self._on_event(kind, False)
                    return None
                self._last_double_t = now
                self._log(f"Double {kind.value} detected")
                if self._on_event:
                    self._on_event(kind, True)
                return kind
            if gap_ms > max_ms:
                # Stale pending event — this one becomes the new first half of a pair.
                self._pending_single[kind] = now
                self._log(f"Single {kind.value} detected")
                if self._on_event:
                    self._on_event(kind, False)
                return None
            # gap_ms < min_ms: too fast to be a deliberate second gesture (likely the
            # same physical clap/snap re-triggering) — ignore, keep the original pending.
            return None

        self._pending_single[kind] = now
        self._log(f"Single {kind.value} detected")
        if self._on_event:
            self._on_event(kind, False)
        return None

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[Gesture] {msg}")


def _spectral_centroid(samples: np.ndarray) -> float:
    """Crude 'brightness' measure: where the energy is concentrated in the
    spectrum. Higher = more high-frequency-dominant (snap-like), lower =
    more broadband/low-frequency (clap-like)."""
    if samples.size < 8:
        return 0.0
    windowed = samples.astype(np.float64) * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / SAMPLE_RATE)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float((freqs * spectrum).sum() / total)
