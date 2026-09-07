# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hold-to-talk — a global key listener.

HOLD the key -> mic opens. RELEASE -> mic closes and the utterance is
processed. The button IS the voice-activity detector, which is why this
mode is speaker-safe with no headphones: the mic simply isn't open while
the assistant talks, unless you press the key — and pressing while it
talks interrupts it.

THE KEY-REPEAT TRAP (the bug that kills every naive build): the OS fires
on_press events CONTINUOUSLY while a key is held. Without the held-state
filter below, every repeat reads as a fresh press and keeps cancelling
the reply before it can speak.

AND THE HALF THAT TRAP HIDES: some keyboards send auto-repeat as full
DOWN/UP PAIRS rather than the repeated DOWN-only stream. Filtering the
presses and trusting every release then breaks the OTHER way -- a single
hold is chopped into dozens of ~50ms recordings, each too short to
transcribe, and the whole thing is SILENT. No exception, no log line,
nothing to search for; it simply reads as "the microphone does not work".
Measured in the field on a Logitech MX Mechanical through a Bolt
receiver: one 2.6-second hold produced 186 key events and about fifty
recordings. So a release is never trusted on sight -- see is_held().

macOS needs Input Monitoring permission for the hosting terminal
(System Settings -> Privacy & Security -> Input Monitoring). Windows
works out of the box.

LINUX / WAYLAND: pynput's global listener is useless here -- its X11
backend gets no events under a Wayland compositor (keystrokes to native
Wayland windows never reach XWayland), and its evdev backend refuses to
start without root. So on Linux we read /dev/input/event* directly with
python-evdev, which is Wayland-agnostic and needs only membership in the
`input` group (the device nodes are `crw-rw---- root input`). Non-Linux
platforms keep the original pynput path unchanged. If evdev import or
device discovery fails, we fall back to pynput so nothing regresses.
"""
import sys
import threading
import time

# ---------------------------------------------------------------------------
# pynput key resolution (used by the pynput fallback path and every non-Linux
# platform). Unchanged from upstream.
# ---------------------------------------------------------------------------
try:
    from pynput import keyboard as _pyn_keyboard
except Exception:                       # pragma: no cover - non-Linux always has it
    _pyn_keyboard = None


def resolve_key(name: str):
    """'home' / 'f13' / 'right_alt' / any single character -> pynput key."""
    keyboard = _pyn_keyboard
    name = (name or "home").strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    # Friendly names -> pynput's names. pynput calls the right option key
    # alt_r, not right_alt; the docs speak human, this map translates.
    # (Field-caught: right_alt silently fell back to home, which Mac
    # laptops cannot press, so the voice looked healthy and never fired.)
    aliases = {
        "right_alt": "alt_r", "left_alt": "alt_l",
        "right_option": "alt_r", "left_option": "alt_l",
        "right_ctrl": "ctrl_r", "left_ctrl": "ctrl_l",
        "right_cmd": "cmd_r", "left_cmd": "cmd_l",
        "right_shift": "shift_r", "left_shift": "shift_l",
    }
    name = aliases.get(name, name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return keyboard.Key.home


# ---------------------------------------------------------------------------
# evdev key resolution (Linux). Maps the same friendly names to Linux input
# event codes (KEY_*).
# ---------------------------------------------------------------------------
def _resolve_evdev_key(name: str) -> int:
    from evdev import ecodes
    raw = (name or "home").strip().lower()

    aliases = {
        "right_alt": "rightalt", "left_alt": "leftalt",
        "right_option": "rightalt", "left_option": "leftalt",
        "right_ctrl": "rightctrl", "left_ctrl": "leftctrl",
        "right_cmd": "rightmeta", "left_cmd": "leftmeta",
        "right_meta": "rightmeta", "left_meta": "leftmeta",
        "right_shift": "rightshift", "left_shift": "leftshift",
        "page_up": "pageup", "page_down": "pagedown",
        "pgup": "pageup", "pgdn": "pagedown",
        "del": "delete", "ins": "insert", "esc": "escape",
        "return": "enter", "caps_lock": "capslock",
        "menu": "compose", "printscreen": "sysrq", "print_screen": "sysrq",
    }
    key = aliases.get(raw, raw)
    candidates = ["KEY_" + key.upper().replace("-", "").replace("_", "")]
    if len(raw) == 1:
        candidates.insert(0, "KEY_" + raw.upper())

    for cand in candidates:
        code = ecodes.ecodes.get(cand)
        if code is not None:
            return code
    print(f"[ptt] unknown key {name!r} for evdev — falling back to 'home'",
          flush=True)
    return ecodes.ecodes["KEY_HOME"]


def _evdev_keyboards(target_code: int):
    """Devices that could emit `target_code`, most specific first.

    Prefer devices that actually advertise the key; then any real keyboard
    (advertises ENTER); then anything with key events at all. A key can fire
    without being listed in capabilities on some drivers, so the broader
    tiers are a genuine safety net, not dead code.
    """
    import evdev
    from evdev import ecodes

    advertised, keyboards, any_keys = [], [], []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        keys = dev.capabilities().get(ecodes.EV_KEY, [])
        if not keys:
            dev.close()
            continue
        if target_code in keys:
            advertised.append(dev)
        elif ecodes.KEY_ENTER in keys:
            keyboards.append(dev)
        else:
            any_keys.append(dev)

    chosen = advertised or keyboards or any_keys
    for dev in (advertised + keyboards + any_keys):
        if dev not in chosen:
            dev.close()
    return chosen


class PTTListener:
    # How long a release must stand unchallenged before it is believed.
    # Comfortably longer than any keyboard's auto-repeat period (measured
    # at ~50ms on the hardware that exposed this; Windows' fastest setting
    # is ~30ms) and short enough that letting go still feels instant.
    RELEASE_GRACE = 0.12

    def __init__(self, key="home"):
        self._raw_key = key
        self._held = False
        self._release_t = None          # a release awaiting confirmation
        self._press_evt = threading.Event()
        self._threads = []

        self._backend = None
        if sys.platform.startswith("linux"):
            try:
                self._start_evdev(key)
                self._backend = "evdev"
            except Exception as e:      # import error, no devices, no perms
                print(f"[ptt] evdev backend unavailable ({e}); "
                      f"falling back to pynput", flush=True)

        if self._backend is None:
            self._start_pynput(key)
            self._backend = "pynput"

    # -- shared state transitions (source-agnostic) -------------------------
    def _key_down(self):
        # A press cancels any pending release: that release was auto-repeat,
        # not a human letting go.
        self._release_t = None
        if not self._held:                      # filter key-repeat
            self._held = True
            self._press_evt.set()

    def _key_up(self):
        # PROVISIONAL. Believed only if no press follows; see _settle().
        self._release_t = time.monotonic()

    # -- evdev backend (Linux) -------------------------------------------------
    def _start_evdev(self, key):
        import evdev  # noqa: F401  (probe the import before committing)

        self._evdev_code = _resolve_evdev_key(key) if isinstance(key, str) else key
        devices = _evdev_keyboards(self._evdev_code)
        if not devices:
            raise RuntimeError("no readable input devices in /dev/input "
                               "(is the user in the `input` group?)")

        names = ", ".join(sorted({d.name for d in devices}))
        print(f"[ptt] evdev: watching key code {self._evdev_code} on "
              f"{len(devices)} device(s): {names}", flush=True)

        for dev in devices:
            t = threading.Thread(target=self._evdev_loop, args=(dev,),
                                 daemon=True)
            t.start()
            self._threads.append(t)

    def _evdev_loop(self, dev):
        from evdev import ecodes
        try:
            for event in dev.read_loop():
                if event.type != ecodes.EV_KEY or event.code != self._evdev_code:
                    continue
                if event.value == 1:            # key down
                    self._key_down()
                elif event.value == 0:          # key up
                    self._key_up()
                # event.value == 2 is auto-repeat; ignore it outright.
        except OSError:
            # Device unplugged mid-session. The other watcher threads (and a
            # reconnect on the next launch) carry on; nothing to recover here.
            print(f"[ptt] evdev: lost device {dev.path} ({dev.name})",
                  flush=True)
        finally:
            try:
                dev.close()
            except Exception:
                pass

    # -- pynput backend (macOS, Windows, Linux fallback) --------------------
    def _start_pynput(self, key):
        keyboard = _pyn_keyboard
        self._pyn_key = resolve_key(key) if isinstance(key, str) else key
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, k):
        if k == self._pyn_key:
            self._key_down()

    def _on_release(self, k):
        if k == self._pyn_key:
            self._key_up()

    # -- release settling + public API ------------------------------------------
    def _settle(self):
        """Commit a release that has stood unchallenged for the grace window."""
        r = self._release_t
        if self._held and r is not None and \
                time.monotonic() - r >= self.RELEASE_GRACE:
            self._held = False
            self._release_t = None

    def wait_press(self):
        """Block until the key goes DOWN (one event per physical press)."""
        # Settled on a loop, not once. A release landing after the last
        # is_held() poll leaves _held provisionally True, and a single
        # settle-then-wait would then block forever: the next press is
        # filtered as key-repeat, so nothing ever sets the event again.
        while True:
            self._settle()
            if self._press_evt.wait(timeout=self.RELEASE_GRACE):
                self._press_evt.clear()
                return

    def is_held(self) -> bool:
        self._settle()
        return self._held
