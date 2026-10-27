"""Low-level Windows hardware scan-code keyboard simulator for games."""

import ctypes
import time
from typing import Dict

# Windows SendInput constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001

# Hardware DirectInput / Scan codes for arrow and WASD keys:
# Standard Subway Surfers on PC / MS Store responds to arrow keys, WASD, and Space.
SCAN_CODES: Dict[str, int] = {
    "left": 0x4B,       # Left arrow (extended)
    "right": 0x4D,      # Right arrow (extended)
    "up": 0x48,         # Up arrow (jump / extended)
    "down": 0x50,       # Down arrow (roll / slide / extended)
    "a": 0x1E,
    "d": 0x20,
    "w": 0x11,
    "s": 0x1F,
    "space": 0x39,
}

EXTENDED_KEYS = {"left", "right", "up", "down"}

# Ctypes struct definitions for SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput),
    ]


class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I),
    ]


SendInput = ctypes.windll.user32.SendInput


def press_key(key_name: str) -> None:
    """Send key down event using hardware scan code."""
    scan = SCAN_CODES.get(key_name.lower())
    if scan is None:
        return
    flags = KEYEVENTF_SCANCODE
    if key_name.lower() in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, scan, flags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(key_name: str) -> None:
    """Send key up event using hardware scan code."""
    scan = SCAN_CODES.get(key_name.lower())
    if scan is None:
        return
    flags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
    if key_name.lower() in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, scan, flags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def tap_key(key_name: str, duration: float = 0.05) -> None:
    """Press and release a key with a small hold time."""
    press_key(key_name)
    time.sleep(duration)
    release_key(key_name)
