import time, os
import pyrealsense2 as rs
import yaml
import numpy as np


def start_rs(pipe, cfg, warmup=15):
    """Start pipeline and discard a few frames to let auto-exposure/USB settle."""
    profile = pipe.start(cfg)
    # Warmup: swallow a few frames; ignore timeouts
    for _ in range(max(0, warmup)):
        try:
            pipe.wait_for_frames(1000)
        except Exception:
            pass
    return profile

def get_frames_with_retry(pipe, cfg, max_retries=2, timeout_ms=2000, do_hw_reset=True):
    """Wait for frames; on timeout restart pipeline (and optionally HW reset)."""
    for attempt in range(max_retries + 1):
        try:
            return pipe.wait_for_frames(timeout_ms)
        except Exception as e:
            if attempt == max_retries:
                raise
            # restart pipeline cleanly
            try:
                pipe.stop()
            except Exception:
                pass
            # optional hardware reset between attempts
            if do_hw_reset:
                try:
                    dev = rs.context().query_devices()[0]
                    dev.hardware_reset()
                    time.sleep(2.0)  # give firmware time to come back
                except Exception:
                    time.sleep(0.5)
            time.sleep(0.5)
            pipe.start(cfg)
            # brief warmup after restart
            for _ in range(5):
                try:
                    pipe.wait_for_frames(500)
                except Exception:
                    pass

def _face_label(face_entry, max_chars=28, default="unregistered"):
    """Short label for overlay: YAML basename, truncated; safe if face_entry is None."""
    if not face_entry:
        return default
    yml = face_entry.get("yaml", "")
    base = os.path.splitext(os.path.basename(yml))[0] if yml else default
    if len(base) > max_chars:
        base = base[:max_chars - 1] + "…"
    return base



def load_calib(path):
    y = yaml.safe_load(open(path, "r"))
    cm = y["camera_matrix"]
    fx, fy, cx, cy = float(cm["fx"]), float(cm["fy"]), float(cm["cx"]), float(cm["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    return (fx, fy, cx, cy), K