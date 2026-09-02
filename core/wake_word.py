"""
core/wake_word.py — Local-first wake-word and stop-phrase detection for SHADOW.

Everything here runs fully offline via Vosk. No microphone audio is ever
sent to a cloud AI/STT provider merely to detect "wake up shadow" or
"stop shadow" — while SHADOW is SLEEPING or STOPPED, the only thing
touching the microphone is this module.

Grammar-constrained recognition is what makes this resistant to false
activation: Vosk is restricted to output ONLY the configured phrase(s) or
the catch-all "[unk]" token, so background chatter, music, claps, snaps
and unrelated speech cannot produce a match — there is no substring/keyword
list to accidentally trigger on.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

import sounddevice as sd

SAMPLE_RATE = 16000
_BLOCK      = 4000   # ~0.25s per read at 16kHz

_model_lock   = threading.Lock()
_shared_model = None  # cached vosk.Model — loaded once, shared by wake + stop listeners


def _get_model():
    """Lazily load (and cache) the local Vosk model. Downloads once on first
    use if not already cached (~50MB, small English model), then runs fully
    offline for every session after that."""
    global _shared_model
    with _model_lock:
        if _shared_model is None:
            from vosk import Model
            print("[WakeWord] Loading local offline speech model (Vosk)…")
            _shared_model = Model(lang="en-us")
            print("[WakeWord] Local wake-word engine ready.")
        return _shared_model


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class LocalPhraseListener:
    """
    Grammar-constrained local phrase detector.

    Two usage modes:
      - listen_blocking(stop_flag): owns the microphone (opens its own
        InputStream) and blocks the calling thread until one of `phrases`
        is recognized or `stop_flag` is set. Used while SLEEPING — this is
        the ONLY thing allowed to touch the mic in that state (mic mode
        WAKE_WORD_ONLY).
      - feed(pcm_bytes): does not own a stream; processes externally
        supplied raw int16 mono PCM at SAMPLE_RATE. Used while ACTIVE to
        watch for "stop shadow" on the audio already being captured for
        the conversation, without opening a second competing input stream.
    """

    def __init__(self, phrases: list[str], cooldown_s: float = 1.5):
        self._phrases      = [_normalize(p) for p in phrases if p and p.strip()]
        self._cooldown_s   = cooldown_s
        self._last_match_t = 0.0
        self._recognizer    = None
        self._lock          = threading.Lock()
        self._init_error: Optional[Exception] = None

    def _ensure_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer
        from vosk import KaldiRecognizer
        model    = _get_model()
        grammar  = json.dumps(self._phrases + ["[unk]"])
        rec      = KaldiRecognizer(model, SAMPLE_RATE, grammar)
        rec.SetWords(False)
        self._recognizer = rec
        return rec

    def _check_text(self, text: str) -> Optional[str]:
        text = _normalize(text)
        if not text:
            return None
        for phrase in self._phrases:
            if text == phrase:
                now = time.monotonic()
                if now - self._last_match_t < self._cooldown_s:
                    return None   # debounce: ignore immediate repeats
                self._last_match_t = now
                return phrase
        return None

    def feed(self, pcm_bytes: bytes) -> Optional[str]:
        """Feed raw int16 mono PCM @16kHz. Returns the matched phrase, or None."""
        with self._lock:
            try:
                rec = self._ensure_recognizer()
            except Exception as e:
                self._init_error = e
                return None
            if rec.AcceptWaveform(pcm_bytes):
                result = json.loads(rec.Result())
                return self._check_text(result.get("text", ""))
        return None

    def listen_blocking(self, stop_flag: threading.Event) -> Optional[str]:
        """
        Opens the microphone (WAKE_WORD_ONLY — nothing else reads it while
        this runs) and blocks until a configured phrase matches or
        `stop_flag` is set externally. Returns the matched phrase, or None
        if cancelled without a match.
        """
        with self._lock:
            self._recognizer = None
            try:
                self._ensure_recognizer()
            except Exception as e:
                self._init_error = e
                raise

        matched: list[Optional[str]] = [None]

        def _callback(indata, frames, time_info, status):
            if matched[0] is not None or stop_flag.is_set():
                return
            m = self.feed(bytes(indata))
            if m:
                matched[0] = m

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=_BLOCK, callback=_callback,
        ):
            while matched[0] is None and not stop_flag.is_set():
                time.sleep(0.05)

        return matched[0]
