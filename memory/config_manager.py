import json
import sys
from pathlib import Path

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR    = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def config_exists() -> bool:
    return CONFIG_FILE.exists()

def save_api_keys(gemini_api_key: str) -> None:
    ensure_config_dir()

    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data["gemini_api_key"] = gemini_api_key.strip()

    CONFIG_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

def load_api_keys() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return {}

def get_gemini_key() -> str | None:
    return load_api_keys().get("gemini_api_key")

def is_configured() -> bool:
    key = get_gemini_key()
    return bool(key and len(key) > 15)


def get_assistant_name() -> str:
    """Return the configured assistant name, or 'SHADOW' if not set."""
    return load_api_keys().get("assistant_name", "SHADOW") or "SHADOW"


def get_user_name() -> str:
    """Return the configured user name for addressing."""
    return load_api_keys().get("user_name", "")


DEFAULT_WAKE_PHRASE = "wake up shadow"
DEFAULT_STOP_PHRASE = "stop shadow"


def get_wake_phrase() -> str:
    """Return the configured wake phrase (normalized lowercase)."""
    return (load_api_keys().get("wake_phrase") or DEFAULT_WAKE_PHRASE).strip().lower()


def get_stop_phrase() -> str:
    """Return the configured stop phrase (normalized lowercase)."""
    return (load_api_keys().get("stop_phrase") or DEFAULT_STOP_PHRASE).strip().lower()


# Maps the config-file keys (matching the names in the spec) to GestureConfig
# dataclass field names. Defaults live in ONE place — core.gesture.GestureConfig
# — this only supplies overrides, so thresholds are never hardcoded here or
# duplicated between files.
_GESTURE_CONFIG_KEYS = {
    "DOUBLE_CLAP_ENABLED":           "enabled_claps",
    "DOUBLE_SNAP_ENABLED":           "enabled_snaps",
    "CLAP_DETECTION_THRESHOLD":      "clap_detection_threshold",
    "SNAP_DETECTION_THRESHOLD":      "snap_detection_threshold",
    "DOUBLE_CLAP_MIN_INTERVAL_MS":   "double_clap_min_interval_ms",
    "DOUBLE_CLAP_MAX_INTERVAL_MS":   "double_clap_max_interval_ms",
    "DOUBLE_SNAP_MIN_INTERVAL_MS":   "double_snap_min_interval_ms",
    "DOUBLE_SNAP_MAX_INTERVAL_MS":   "double_snap_max_interval_ms",
    "GESTURE_COOLDOWN_MS":           "gesture_cooldown_ms",
}


def get_gesture_config():
    """
    Build a core.gesture.GestureConfig from config/api_keys.json's optional
    "gesture" object, falling back to the dataclass defaults for anything
    unset. Import is local to avoid a hard dependency on numpy (core.gesture)
    for callers that never touch gesture detection.
    """
    from core.gesture import GestureConfig

    raw = load_api_keys().get("gesture")
    if not isinstance(raw, dict):
        raw = {}
    kwargs = {}
    for env_key, field_name in _GESTURE_CONFIG_KEYS.items():
        if env_key in raw:
            kwargs[field_name] = raw[env_key]
    return GestureConfig(**kwargs)


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist assistant name and user name to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["assistant_name"] = assistant_name.strip() or "SHADOW"
    data["user_name"] = user_name.strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_brief_enabled() -> bool:
    return load_api_keys().get("morning_brief_enabled", True)


def save_brief_enabled(enabled: bool) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["morning_brief_enabled"] = enabled
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_plugin_enabled(plugin_name: str) -> bool:
    """Plugins are enabled by default the moment they're discovered (opt-out model)."""
    return load_api_keys().get("plugins_enabled", {}).get(plugin_name, True)


def save_plugin_enabled(plugin_name: str, enabled: bool) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    plugins_cfg = data.get("plugins_enabled")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
    plugins_cfg[plugin_name] = enabled
    data["plugins_enabled"] = plugins_cfg
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")