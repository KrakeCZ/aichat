"""Read-only ADB hardware dashboard. Python 3.10+, standard library."""
import argparse
from collections import deque
import csv
from datetime import datetime
from decimal import Decimal
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BACKENDS = ("CPU", "GPU", "Vulkan", "OpenCL", "OpenGL", "NPU",
            "QNN", "QNN-HTP", "QNN-GPU", "QNN-CPU", "QNN-DSP", "DSP")
DEFAULT_BACKENDS = ("CPU", "GPU", "Vulkan", "OpenCL", "OpenGL", "NPU", "QNN-HTP", "DSP")
DEFAULT_TEMPS = tuple(f"{key} · teplota" for key in ("CPU", "GPU", "NPU", "Baterie", "CDSP"))
THERMAL_STATES = ("bez omezení", "mírný", "střední", "silný", "kritický", "nouzový", "vypnutí")
PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+(?::[A-Za-z0-9_.]+)?")
BG, PANEL, EDGE = "#10151e", "#19222f", "#2a394c"
TEXT, MUTED, MINT = "#f0f5ff", "#a7bacf", "#60e4bd"
COLORS = ("#61e8c0", "#76baff", "#ffc477", "#d7aaff", "#ff8496", "#e0ef8b")
HISTORY = 120
CANCELLED = object()

# Fixed, read-only scripts are sent through stdin, outside Windows/shell argument quoting.
HARDWARE = r'''
export LC_ALL=C
printf '@cpu\n'
grep '^cpu' /proc/stat 2>/dev/null
printf '@cpufreq\n'
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
  if read -r v 2>/dev/null < "$f"; then
    core=${f%/cpufreq/*}
    printf '%s\t%s\n' "${core##*/}" "$v"
  fi
done
printf '@gpu_percent\n'
gpu=/sys/class/kgsl/kgsl-3d0
if ! cat "$gpu/gpu_busy_percentage" 2>/dev/null; then
  printf '@gpu_busy\n'
  cat "$gpu/gpubusy" 2>/dev/null
fi
printf '@gpufreq\n'
cat "$gpu/gpuclk" 2>/dev/null
printf '@devfreq\n'
for d in /sys/class/devfreq/*; do
  name=$(
    if read -r n 2>/dev/null < "$d/name"; then printf '%s' "$n"; fi
    printf ' %s' "${d##*/}"
    if [ -L "$d" ]; then printf ' %s' "$(readlink "$d")"; fi
  )
  case "$name" in
    *npu*|*nsp*|*cdsp*|*NPU*|*NSP*|*CDSP*)
      if read -r v 2>/dev/null < "$d/cur_freq"; then
        printf '%s\t%s\t%s\n' "${d##*/}" "$name" "$v"
      fi ;;
  esac
done
printf '@end\n'
'''

THERMAL = r'''
export LC_ALL=C
printf '@zones\n'
for z in /sys/class/thermal/thermal_zone*; do
  if read -r n 2>/dev/null < "$z/type" && read -r v 2>/dev/null < "$z/temp"; then
    printf '%s\t%s\t%s\n' "${z##*/}" "$n" "$v"
  fi
done
printf '@thermalservice\n'
dumpsys -t 2 thermalservice 2>/dev/null
printf '@battery\n'
dumpsys -t 2 battery 2>/dev/null
printf '@end\n'
'''

APPLICATION = r'''
export LC_ALL=C
pkg='__PACKAGE__'
process='__PROCESS__'
printf '@pid\n'
pids=$(pidof "$process" 2>/dev/null)
pid=${pids%% *}
printf '%s\n' "$pid"
case "$pid" in ''|*[!0-9]*) printf '@end\n'; exit 0 ;; esac
printf '@mem\n'
dumpsys -t 2 meminfo "$pid" 2>/dev/null
printf '@maps\n'
run-as "$pkg" cat "/proc/$pid/maps" 2>/dev/null | grep -iE '/[^ /]*(vulkan|opencl|libEGL|libGLES|qnn|hexagon|cdsprpc|adsprpc|fastrpc)[^ /]*\.so' | head -n 250
printf '@maps_status\n'
run-as "$pkg" sh -c 'test -r "/proc/$1/maps" && head -c 1 "/proc/$1/maps" >/dev/null' sh "$pid" 2>/dev/null && printf 'readable\n'
printf '@processstat\n'
cat "/proc/$pid/stat" 2>/dev/null || run-as "$pkg" cat "/proc/$pid/stat" 2>/dev/null
printf '@totalstat\n'
head -n 1 /proc/stat 2>/dev/null
printf '@end\n'
'''

ACTIVITY = r'''
export LC_ALL=C
process='__PROCESS__'
printf '@pid\n'
pids=$(pidof "$process" 2>/dev/null)
pid=${pids%% *}
printf '%s\n' "$pid"
case "$pid" in ''|*[!0-9]*) printf '@end\n'; exit 0 ;; esac
printf '@timing\n'
logcat -b main -d -t 2000 --pid="$pid" -v threadtime -v epoch -v usec ADBTelemetry:I '*:S' 2>/dev/null
printf '@logstatus\n%s\n' "$?"
printf '@end\n'
'''

LIBRARIES = {
    "Vulkan": re.compile(r"(?:libvulkan|vulkan\.)[^/]*\.so(?:$|\.)", re.I),
    "OpenCL": re.compile(r"(?:libopencl|opencl\.)[^/]*\.so(?:$|\.)", re.I),
    "OpenGL": re.compile(r"lib(?:egl|gles)[^/]*\.so(?:$|\.)", re.I),
    "NPU": re.compile(r"libqnnhtp[^/]*\.so(?:$|\.)", re.I),
    "QNN": re.compile(r"libqnn[^/]*\.so(?:$|\.)", re.I),
    "QNN-HTP": re.compile(r"libqnnhtp[^/]*\.so(?:$|\.)", re.I),
    "QNN-GPU": re.compile(r"libqnngpu[^/]*\.so(?:$|\.)", re.I),
    "QNN-CPU": re.compile(r"libqnncpu[^/]*\.so(?:$|\.)", re.I),
    "QNN-DSP": re.compile(r"libqnndsp[^/]*\.so(?:$|\.)", re.I),
    "DSP": re.compile(r"lib(?:qnndsp|cdsprpc|adsprpc|fastrpc|hexagon)[^/]*\.so(?:$|\.)", re.I),
}
LOGCAT_LINE = re.compile(
    r"^\s*(\d{1,15}\.\d{1,9})\s+(\d+)\s+\d+\s+[A-Z]\s+ADBTelemetry\s*:\s*(.{1,300})$"
)
OP_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
RUN_TIMEOUT = 45


def adb_command(executable, args, cancel, script=None, timeout=8):
    if cancel.is_set():
        return CANCELLED
    with subprocess.Popen(
        [executable, *args],
        stdin=subprocess.PIPE if script is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as proc:
        payload = script.encode("utf-8") if script is not None else None
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set() or time.monotonic() >= deadline:
                proc.kill()
                proc.communicate()
                if cancel.is_set():
                    return CANCELLED
                raise RuntimeError(f"ADB neodpovědělo do {timeout} s.")
            try:
                stdout, stderr = proc.communicate(payload, timeout=0.15)
            except subprocess.TimeoutExpired:
                payload = None
                continue
            if proc.returncode:
                detail = (stderr or stdout).decode("utf-8", "replace").strip()
                raise RuntimeError((detail or f"ADB skončilo s kódem {proc.returncode}.")[:550])
            return stdout.decode("utf-8", "replace")


def sections(raw):
    result = {}
    target = None
    for line in raw.splitlines():
        if re.fullmatch(r"@[a-z_]+", line):
            target = result.setdefault(line[1:], [])
        elif target is not None:
            target.append(line)
    if "end" not in result:
        raise RuntimeError("ADB vrátilo neúplný vzorek.")
    return result


def number(text):
    if re.fullmatch(r"-?\d{1,15}(?:\.\d{1,8})?", text.strip()):
        return float(text)
    return None


def observation(value, unit, source):
    return value, unit, source


class Hardware:
    def __init__(self):
        self.previous = {}

    def read(self, s):
        data, current = {}, {}
        for line in s.get("cpu", []):
            fields = line.split()
            if len(fields) < 5 or not re.fullmatch(r"cpu\d*", fields[0]):
                continue
            key = fields[0]
            try:
                # guest and guest_nice are already included in user and nice.
                ticks = list(map(int, fields[1:9]))
            except ValueError:
                continue
            idle = sum(ticks[3:5])
            total = sum(ticks)
            current[key] = total, idle
            previous = self.previous.get(key)
            if previous:
                dt, di = total - previous[0], idle - previous[1]
                if dt > 0 and 0 <= di <= dt:
                    name = "CPU" if key == "cpu" else f"CPU {key[3:]}"
                    data[f"{name} · zátěž"] = observation(100 * (dt - di) / dt, "%", f"/proc/stat {key}")
        self.previous = current
        for line in s.get("cpufreq", []):
            key, sep, raw = line.partition("\t")
            value = number(raw)
            if sep and re.fullmatch(r"cpu\d+", key) and value is not None and value > 0:
                data[f"CPU {key[3:]} · takt"] = observation(value / 1000, "MHz", f"{key}/cpufreq/scaling_cur_freq")

        for line in s.get("gpu_percent", []):
            m = re.fullmatch(r"\s*(\d+)\s*%?\s*", line)
            if m and int(m[1]) <= 100:
                data["GPU · zátěž"] = observation(float(m[1]), "%", "KGSL gpu_busy_percentage")
        if "GPU · zátěž" not in data:
            for line in s.get("gpu_busy", []):
                m = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", line)
                if m and 0 <= int(m[1]) <= int(m[2]) and int(m[2]) > 0:
                    data["GPU · zátěž"] = observation(100 * int(m[1]) / int(m[2]), "%", "KGSL gpubusy")
        for line in s.get("gpufreq", []):
            hz = number(line)
            if hz is not None and hz > 0:
                data["GPU · takt"] = observation(hz / 1e6, "MHz", "KGSL gpuclk")
        for line in s.get("devfreq", []):
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            node, label, raw = parts
            # Interconnect/DDR devices can also have cdsp or npu in their name.
            if re.search(r"(?:bw|ddr|llcc|icc|interconnect|bus)", label, re.I):
                continue
            hz = number(raw)
            if hz is not None and 0 < hz <= 10e9:
                data[f"Devfreq · {node}"] = observation(hz / 1e6, "MHz", f"/sys/class/devfreq/{node}/cur_freq")
        return data


def thermal(s):
    data, groups = {}, {}
    for line in s.get("zones", []):
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        zone, label, raw = parts
        value = number(raw)
        if value is not None and -40000 <= value <= 200000:
            groups[(f"{label} ({zone})", label, None)] = (value / 1000, f"{zone}/temp")
    active = None
    for line in s.get("thermalservice", []):
        status = re.fullmatch(r"\s*Thermal Status:\s*([0-6])\s*", line, re.I)
        if status:
            level = int(status[1])
            data["Android · tepelný stav"] = observation(level, "stupeň", "dumpsys thermalservice")
        heading = line.strip().casefold()
        if heading.startswith("current temperatures"):
            active = True
        elif heading.endswith(":"):
            active = False
        if active is False:
            continue
        for row in re.findall(r"Temperature\{([^}]+)\}", line):
            fields = dict(re.findall(r"(m\w+)=([^,]+)", row))
            label = fields.get("mName", "").strip()
            value = number(fields.get("mValue", ""))
            kind = fields.get("mType", "").strip()
            if label and kind not in {"6", "7", "8"} and value is not None and -40 <= value <= 200:
                groups[(f"{label} (Android)", label, kind)] = (value, "thermalservice")
    candidates = {}
    for (name, label, kind), (value, source) in groups.items():
        data[f"Senzor · {name}"] = observation(value, "°C", source)
        family = {"0": "CPU", "1": "GPU", "2": "Baterie", "9": "NPU"}.get(kind)
        if family is None and kind in (None, "", "-1"):
            lower = label.casefold()
            for category, pattern in (
                ("GPU", r"gpu|adreno"), ("NPU", r"npu|(?:^|[^a-z])nsp(?:[^a-z]|$)"),
                ("CDSP", r"cdsp|hexagon"), ("CPU", r"cpu|apss"), ("Baterie", r"batt"),
            ):
                if re.search(pattern, lower):
                    family = category
                    break
        if family and (family not in candidates or candidates[family][0] < value):
            candidates[family] = value, f"{source} · {name}"
    for line in s.get("battery", []):
        match = re.fullmatch(r"\s*temperature:\s*(-?\d+)\s*", line)
        if match and -400 <= int(match[1]) <= 1500:
            candidates["Baterie"] = int(match[1]) / 10, "dumpsys battery"
    for name, (value, source) in candidates.items():
        data[f"{name} · teplota"] = observation(value, "°C", source)
    return data


class AppMetrics:
    def __init__(self):
        self.cpu_sample = None

    def process_cpu(self, s, pid):
        line = next(iter(s.get("processstat", [])), "")
        _, separator, tail = line.rpartition(") ")
        total = next(iter(s.get("totalstat", [])), "").split()
        fields = tail.split()
        current = None
        if separator and line.startswith(pid + " (") and len(fields) > 19 and len(total) >= 5 and total[0] == "cpu":
            try:
                current = (pid, int(fields[19]), int(fields[11]) + int(fields[12]), sum(map(int, total[1:9])))
            except ValueError:
                pass
        previous, self.cpu_sample = self.cpu_sample, current
        if current and previous and current[:2] == previous[:2]:
            elapsed, busy = current[3] - previous[3], current[2] - previous[2]
            if elapsed > 0 and 0 <= busy <= elapsed:
                return 100 * busy / elapsed
        return None

    def read(self, s):
        data = {}
        pid = next(iter(s.get("pid", [])), "").strip()
        if not pid.isdecimal():
            self.cpu_sample = None
            return data, "Proces aplikace neběží nebo jej ADB nevidí."
        cpu = self.process_cpu(s, pid)
        if cpu is not None:
            data["CPU procesu · zátěž"] = observation(cpu, "%", f"/proc/{pid}/stat vůči kapacitě všech CPU")
        for line in s.get("mem", []):
            match = re.fullmatch(r"\s*(Gfx dev|EGL mtrack|GL mtrack|Native Heap)\s+(\d+)(?:\s+\d+){2,}\s*", line)
            if match:
                data[match[1]] = observation(int(match[2]) / 1024, "MiB", f"dumpsys meminfo PID {pid} / PSS")
        readable = "readable" in s.get("maps_status", [])
        names = set()
        for line in s.get("maps", []):
            match = re.search(r"/([^/\s]+\.so(?:\.[0-9]+)*)(?:\s+\(deleted\))?\s*$", line)
            if match:
                names.add(match[1])
        for backend, pattern in LIBRARIES.items():
            matched = sorted(name for name in names if pattern.fullmatch(name))
            if matched or readable:
                data[f"{backend} · knihovna"] = observation(
                    int(bool(matched)), "stav", ", ".join(matched) or f"run-as /proc/{pid}/maps"
                )
        status = f"PID {pid} · " + ("mapy čitelné" if readable else "mapy nepřístupné (run-as)")
        return data, status


def activity_event(payload):
    pairs = payload.split()
    if any(not re.fullmatch(r"[a-z]+=[A-Za-z0-9_.-]+", pair) for pair in pairs):
        return None
    fields = dict(pair.split("=", 1) for pair in pairs)
    if len(fields) != len(pairs) or fields.get("backend") not in BACKENDS:
        return None
    backend = fields["backend"]
    kind = fields.get("event", "legacy")
    operation = fields.get("id", "")
    proof = fields.get("proof", "runtime")
    ms = number(fields.get("ms", ""))
    target = fields.get("to")
    if kind == "legacy":
        if set(fields) != {"backend", "ms"} or ms is None or not 0 <= ms <= 3_600_000:
            return None
    elif not OP_ID.fullmatch(operation):
        return None
    elif kind == "done":
        if proof not in ("runtime", "profile") or (ms is not None and not 0 <= ms <= 3_600_000):
            return None
        if "ms" in fields and ms is None:
            return None
    elif kind == "fallback":
        if target not in BACKENDS or target == backend:
            return None
    elif kind not in ("start", "beat", "error"):
        return None
    return backend, kind, operation, proof, ms, target


class BackendActivity:
    def __init__(self):
        self.pid = ""
        self.watermark = None
        self.at_watermark = set()
        self.active = {}
        self.last = {}

    def read(self, s, now):
        pid = next(iter(s.get("pid", [])), "").strip()
        if pid != self.pid or not pid.isdecimal():
            self.__init__()
            self.pid = pid if pid.isdecimal() else ""
            baseline = True
        else:
            baseline = False
        if not self.pid:
            return {}, {}, "Čekám na proces."
        if s.get("logstatus") != ["0"]:
            self.__init__()
            return {}, {}, "Logcat není pro vybraný proces dostupný."

        rows = []
        for line in s.get("timing", []):
            match = LOGCAT_LINE.fullmatch(line)
            if match and match[2] == pid:
                rows.append((Decimal(match[1]), match[3]))
        rows.sort(key=lambda row: row[0])
        data = {}
        for stamp, payload in rows:
            if self.watermark is None or stamp > self.watermark:
                self.watermark = stamp
                self.at_watermark.clear()
            elif stamp < self.watermark or payload in self.at_watermark:
                continue
            self.at_watermark.add(payload)
            event = activity_event(payload)
            if event and not baseline:
                self.apply(event, now, data, f"ADBTelemetry PID {pid} · {stamp}")
        self.active = {key: seen for key, seen in self.active.items() if now - seen < RUN_TIMEOUT}
        return data, self.snapshot(now), f"PID {pid} · ADBTelemetry"

    def apply(self, event, now, data, source):
        backend, kind, op, proof, ms, target = event
        key = backend, op
        if kind in ("start", "beat"):
            if len(self.active) < 1024 or key in self.active:
                self.active[key] = now
        else:
            self.active.pop(key, None)
        confirmed = kind == "done"
        if kind == "fallback":
            status, detail = "Fallback", f"Směr → {target}"
        elif kind == "error":
            status, detail = "Chyba", "Aplikace hlásí neúspěch"
        elif confirmed:
            status = "Dokončeno · profil" if proof == "profile" else "Dokončeno · runtime"
            detail = "Profil z aplikace" if proof == "profile" else "Potvrzeno aplikací"
        elif kind == "legacy":
            status, detail = "Naměřen čas", "Značka ADBTelemetry v1"
        else:
            status, detail = "Poslední start", "Aplikace hlásila odeslání"
        if kind != "beat":
            self.last[backend] = (now, status, detail if ms is None else f"{detail} · {display(ms)} ms")
            event_name = f"{backend} · {kind}" + (f" → {target}" if target else "")
            count = data.get(event_name, (0,))[0] + 1
            data[event_name] = observation(count, "událost", f"{source} · {proof}")
        if (confirmed or kind == "legacy") and ms is not None:
            data[f"{backend} · čas"] = observation(ms, "ms", f"{source} · {kind}/{proof}")

    def snapshot(self, now):
        result = {}
        for name, (stamp, status, note) in self.last.items():
            age = int(max(0, now - stamp))
            if age >= 15:
                status = "Poslední záznam"
            result[name] = (status, f"{note} · před {age} s")
        for backend, _op in self.active:
            result[backend] = ("Odesláno · čeká", "Aplikace zatím nehlásí dokončení")
        return result


def device_list(raw):
    devices = []
    for line in raw.splitlines():
        m = re.fullmatch(r"(\S+)\s+(device|offline|unauthorized|recovery|sideload|bootloader|no permissions)(?:\s+(.*))?", line)
        if m and not m[1].startswith("*"):
            model = re.search(r"\bmodel:(\S+)", m[3] or "")
            description = (model[1].replace("_", " ") + " · ") if model else ""
            devices.append((m[1], m[2], description + m[1] + "  [" + m[2] + "]"))
    return devices


def find_adb():
    locations = [Path(__file__).resolve().parent / "platform-tools"]
    for env in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        if os.environ.get(env):
            locations.append(Path(os.environ[env]) / "platform-tools")
    if os.environ.get("LOCALAPPDATA"):
        locations.append(Path(os.environ["LOCALAPPDATA"]) / "Android" / "Sdk" / "platform-tools")
    name = "adb.exe" if os.name == "nt" else "adb"
    return shutil.which(name) or next((str(p / name) for p in locations if (p / name).is_file()), "")


def demo_sample(group, seconds):
    data = {}
    for name, (baseline, amplitude, speed, unit) in {
        "CPU · zátěž": (42, 26, .56, "%"), "GPU · zátěž": (58, 32, .42, "%"),
        "GPU · takt": (660, 125, .40, "MHz"), "Devfreq · demo-cdsp": (510, 110, .35, "MHz"),
        "CPU · teplota": (56, 6, .15, "°C"), "GPU · teplota": (49, 4, .20, "°C"),
        "NPU · teplota": (45, 3, .18, "°C"), "Baterie · teplota": (35, 1.6, .12, "°C"),
        "Gfx dev": (56, 10, .36, "MiB"), "EGL mtrack": (24, 5, .24, "MiB"),
        "GL mtrack": (18, 3, .40, "MiB"), "Native Heap": (292, 22, .19, "MiB"),
        "CPU procesu · zátěž": (11, 9, .65, "%"),
    }.items():
        destination = {"°C": "Teploty", "MiB": "Aplikace"}.get(unit, "Hardware")
        if name == "CPU procesu · zátěž":
            destination = "Aplikace"
        if group == destination:
            data[name] = observation(baseline + amplitude * math.sin(seconds * speed), unit, "DEMO")
    for core in range(8):
        if group == "Hardware":
            data[f"CPU {core} · zátěž"] = observation(45 + 35 * math.sin(seconds * .5 + core), "%", "DEMO")
            data[f"CPU {core} · takt"] = observation(1600 + 410 * math.sin(seconds * .3 + core), "MHz", "DEMO")
        if group == "Teploty":
            data[f"Senzor · cpu-{core} (demo)"] = observation(53 + 4 * math.sin(seconds * .15 + core), "°C", "DEMO")
    if group == "Teploty":
        data["Android · tepelný stav"] = observation(1, "stupeň", "DEMO")
    if group == "Aplikace":
        for name in LIBRARIES:
            data[f"{name} · knihovna"] = observation(int(name not in ("OpenCL", "QNN-DSP")), "stav", "DEMO")
    return data


def demo_activity(seconds):
    cycle = int(seconds) % 12
    backend = "Vulkan" if cycle < 6 else "QNN-HTP"
    status = ("Odesláno · čeká", "DEMO · odesláno") if cycle % 6 < 2 else (
        "Dokončeno · profil" if backend == "QNN-HTP" else "Dokončeno · runtime", "DEMO · dokončeno")
    data = {}
    if cycle % 6 >= 2:
        data[f"{backend} · čas"] = observation(20 + 6 * math.sin(seconds), "ms", "DEMO")
    return data, {backend: status}


def sample_worker(events, token, stop, adb, serial, process, group, interval, demo):
    hardware, app, activity = Hardware(), AppMetrics(), BackendActivity()
    script = {"Hardware": HARDWARE, "Teploty": THERMAL, "Aplikace": APPLICATION, "Události": ACTIVITY}[group]
    script = script.replace("__PACKAGE__", process.split(":")[0]).replace("__PROCESS__", process)
    start = time.monotonic()
    while not stop.is_set():
        began = time.monotonic()
        statuses = None
        try:
            if demo:
                if group == "Události":
                    data, statuses = demo_activity(began - start)
                else:
                    data = demo_sample(group, began - start)
                note = "Simulovaná data"
            else:
                raw = adb_command(adb, ["-s", serial, "shell", "-T", "sh"], stop, script)
                if raw is CANCELLED:
                    break
                parsed = sections(raw)
                note = ""
                if group == "Hardware":
                    data = hardware.read(parsed)
                elif group == "Teploty":
                    data = thermal(parsed)
                elif group == "Události":
                    data, statuses, note = activity.read(parsed, time.monotonic())
                else:
                    data, note = app.read(parsed)
            if not stop.is_set():
                events.put((token, "sample", group, (datetime.now().astimezone(), data, note, statuses)))
        except (OSError, ValueError, RuntimeError) as exc:
            hardware.previous.clear()
            app.cpu_sample = None
            activity = BackendActivity()
            if not stop.is_set():
                events.put((token, "error", group, str(exc)))
        stop.wait(max(.15, interval - (time.monotonic() - began)))


def refresh_worker(events, token, stop, adb):
    try:
        raw = adb_command(adb, ["devices", "-l"], stop, timeout=10)
        if raw is not CANCELLED:
            events.put((token, "devices", "", device_list(raw)))
    except (OSError, RuntimeError) as exc:
        events.put((token, "device_error", "", str(exc)))


def display(value, precision=1):
    return "—" if value is None else f"{value:.{precision}f}".replace(".", ",")


class Chart(tk.Canvas):
    def __init__(self, parent, monitor, title, subtitle, unit, signals=(), limit=None):
        super().__init__(parent, background=PANEL, highlightbackground=EDGE, highlightthickness=1)
        self.monitor, self.title, self.subtitle = monitor, title, subtitle
        self.unit, self.signals, self.limit = unit, signals, limit
        self.bind("<Configure>", lambda _event: self.paint())

    def paint(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 210 or height < 120:
            return
        self.label(15, 13, self.title, TEXT, ("Segoe UI", 13, "bold"))
        self.label(15, 37, self.subtitle, MUTED, ("Segoe UI", 9))
        right = self.monitor.ended if self.monitor.ended is not None else time.monotonic()
        span = HISTORY
        if self.monitor.began is not None:
            elapsed = right - self.monitor.began
            span = min(HISTORY, max(30, 30 * math.ceil(elapsed / 30)))
        left = right - span
        all_points = [
            [(t, v) for t, v in self.monitor.history.get(signal, ()) if t >= left]
            for signal, _ in self.signals
        ]
        values = [value for points in all_points for _, value in points if value is not None]
        if self.limit is not None:
            low, high = 0, self.limit
        elif self.unit == "°C":
            low = math.floor((min(values, default=20) - 3) / 10) * 10
            high = max(low + 20, math.ceil((max(values, default=50) + 3) / 10) * 10)
        else:
            low = 0
            top = max(values, default=0) * 1.1
            order = 10 ** math.floor(math.log10(top)) if top > 0 else 1
            high = max(1, math.ceil(top / order) * order)
        x0, x1 = 53, width - 17
        y0 = 60
        # Two columns keep multi-series legends readable at the minimum window width.
        legend_rows = (len(self.signals) + 1) // 2
        y1 = height - 30 - legend_rows * 17
        if y1 < y0 + 12:
            return
        self.label(15, y0 - 2, self.unit, MUTED, ("Segoe UI", 8))
        divisions = 4
        for i in range(divisions + 1):
            y = y1 - i * (y1 - y0) / divisions
            self.create_line(x0, y, x1, y, fill=EDGE)
            n = low + i * (high - low) / divisions
            tick = f"{n:.2f}".rstrip("0").rstrip(".")
            self.create_text(x0 - 5, y, text=tick, anchor="e", fill=MUTED, font=("Segoe UI", 8))
        for division in range(5):
            offset = round(span * (division / 4 - 1))
            x = x1 + (x1 - x0) * (division / 4 - 1)
            self.create_line(x, y0, x, y1, fill=EDGE)
            self.create_text(x, y1 + 6, text=f"{offset} s", anchor="n", fill=MUTED, font=("Segoe UI", 8))
        if not values:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="Bez dostupných dat", fill=MUTED, font=("Segoe UI", 11))
        for i, ((signal, name), points) in enumerate(zip(self.signals, all_points)):
            color = COLORS[i % len(COLORS)]
            prev = None
            for stamp, value in points:
                if value is None:
                    prev = None
                    continue
                x = x0 + (x1 - x0) * min(1, (stamp - left) / span)
                y = y1 - (y1 - y0) * (value - low) / (high - low)
                y = max(y0, min(y1, y))
                if prev is not None and stamp - prev[2] <= 10:
                    self.create_line(prev[0], prev[1], x, y, fill=color, width=2)
                else:
                    self.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
                prev = (x, y, stamp)
            current = self.monitor.current.get(signal)
            value = current[0] if current is not None else None
            shown = display(value)
            self.label(16 + (i % 2) * (width // 2), y1 + 23 + (i // 2) * 17,
                       f"●  {name}  {shown}", color, ("Segoe UI", 9))

    def label(self, x, y, text, color, font):
        self.create_text(x, y, text=text, anchor="nw", fill=color, font=font)


class Monitor:
    def __init__(self, root, demo=False, adb_path=None):
        self.root = root
        self.events = queue.Queue()
        self.token = 0
        self.stop = threading.Event()
        self.running = False
        self.ended = None
        self.began = None
        self.listing = False
        self.workers = []
        self.devices = []
        self.adb = adb_path or find_adb()
        self.demo = tk.BooleanVar(value=demo)
        self.timing = tk.BooleanVar(value=True)
        self.device = tk.StringVar()
        self.package = tk.StringVar(value="aichat.com.llama.example")
        self.period = tk.StringVar(value="1")
        self.connection = tk.StringVar(value="Připraveno")
        self.status = tk.StringVar(value="Vyber zařízení nebo zapni demo.")
        self.app_status = tk.StringVar(value="")
        self.backend_status = {}
        self.visible_backends = list(DEFAULT_BACKENDS)
        self.visible_temps = list(DEFAULT_TEMPS)
        self.timing_enabled = False
        self.history = {}
        self.current = {}
        self.known = {}
        self.notes = {}
        self.charts = []
        self.session = None
        self.count = 0
        self.temp_dir = tempfile.TemporaryDirectory(prefix="adb-telemetry-")
        self.log_path = Path(self.temp_dir.name) / "session.csv"
        self.log = None
        self.details_by_row = {}
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.drain_id = self.root.after(120, self.drain)
        self.redraw_id = self.root.after(600, self.redraw)
        if self.adb and not demo:
            self.refresh()
        else:
            self.controls()

    def build(self):
        root = self.root
        root.title("ADB Hardware Monitor")
        root.geometry("1260x870")
        root.minsize(900, 720)
        root.configure(background=BG)
        root.option_add("*Font", ("Segoe UI", 10))

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, bordercolor=EDGE, fieldbackground=PANEL,
                        lightcolor=EDGE, darkcolor=EDGE, troughcolor=BG, selectbackground="#31536b",
                        selectforeground=TEXT)
        style.configure("TButton", padding=(10, 6), background="#263548")
        style.map("TButton", background=[("disabled", PANEL), ("active", "#37516c")],
                  foreground=[("disabled", "#7e91a8")])
        style.configure("Accent.TButton", background="#28785f", foreground="#ffffff")
        style.map("Accent.TButton", background=[("disabled", PANEL), ("active", "#329474")])
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", padding=(16, 7), background=PANEL)
        style.map("TNotebook.Tab", background=[("selected", "#28425a")], foreground=[("selected", TEXT)])
        style.configure("Treeview", rowheight=25, background=PANEL, fieldbackground=PANEL)
        style.configure("Treeview.Heading", background="#253649", foreground=TEXT)
        style.map("Treeview.Heading", background=[("active", "#314b62")])
        style.map("Treeview", background=[("selected", "#31536b")], foreground=[("selected", TEXT)])
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        root.option_add("*TCombobox*Listbox.background", PANEL)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)

        outer = ttk.Frame(root, padding=(19, 12))
        outer.pack(fill="both", expand=True)
        title = ttk.Frame(outer)
        title.pack(fill="x", pady=(0, 10))
        ttk.Label(title, text="ADB Hardware Monitor", font=("Segoe UI", 24, "bold")).pack(side="left")
        tk.Label(title, textvariable=self.connection, background="#203933", foreground=MINT,
                 padx=12, pady=5, font=("Segoe UI", 10, "bold")).pack(side="right")
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 7))
        ttk.Label(bar, text="Zařízení").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.device_box = ttk.Combobox(bar, textvariable=self.device, state="readonly", width=25)
        self.device_box.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.device_box.bind("<<ComboboxSelected>>", lambda _event: self.controls())
        self.refresh_button = ttk.Button(bar, text="Obnovit", command=self.refresh)
        self.refresh_button.grid(row=0, column=2, padx=3)
        self.browse_button = ttk.Button(bar, text="Vybrat ADB…", command=self.browse)
        self.browse_button.grid(row=0, column=3, padx=3)
        self.demo_check = ttk.Checkbutton(bar, text="Demo", variable=self.demo, command=self.toggle_demo)
        self.demo_check.grid(row=0, column=4, padx=8)
        self.start_button = ttk.Button(bar, text="Spustit", style="Accent.TButton", command=self.toggle)
        self.start_button.grid(row=0, column=5, padx=3)
        self.export_button = ttk.Button(bar, text="Export CSV…", command=self.export)
        self.export_button.grid(row=0, column=6, padx=(3, 0))
        bar.columnconfigure(1, weight=1)

        settings = ttk.Frame(outer)
        settings.pack(fill="x", pady=(0, 10))
        ttk.Label(settings, text="Proces aplikace").pack(side="left", padx=(0, 8))
        self.package_box = ttk.Entry(settings, textvariable=self.package, width=28)
        self.package_box.pack(side="left", padx=(0, 14))
        ttk.Label(settings, text="Interval HW (s)").pack(side="left", padx=(0, 6))
        self.period_box = ttk.Combobox(settings, textvariable=self.period, values=("1", "2", "5"), state="readonly", width=3)
        self.period_box.pack(side="left", padx=(0, 14))
        self.timing_check = ttk.Checkbutton(settings, text="Živé události aplikace (ADBTelemetry)", variable=self.timing)
        self.timing_check.pack(side="left")

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.build_dashboard()
        overview = self.page("Hardware a paměť")
        self.chart(overview, 0, 0, "Zátěž CPU a Adreno", "Celé zařízení · /proc/stat a KGSL", "%",
                   (("CPU · zátěž", "CPU"), ("GPU · zátěž", "GPU")), 100)
        self.chart(overview, 0, 1, "Teploty zařízení", "Nejteplejší dostupný senzor podle kategorie", "°C",
                   tuple((f"{k} · teplota", k) for k in ("CPU", "GPU", "NPU", "Baterie", "CDSP")))
        self.chart(overview, 1, 0, "Paměť aplikace", "PSS vybraného procesu · kategorie Androidu", "MiB",
                   tuple((k, k) for k in ("Native Heap", "Gfx dev", "EGL mtrack", "GL mtrack")))
        self.freq_chart = self.chart(overview, 1, 1, "Takt GPU a NPU / CDSP", "Čitelné uzly KGSL a devfreq", "MHz",
                                     (("GPU · takt", "GPU"),))

        self.details = ttk.Frame(self.tabs, padding=(3, 6, 3, 0))
        self.tabs.add(self.details, text="Jádra a senzory")
        self.details.columnconfigure(0, weight=1)
        self.details.columnconfigure(1, weight=1)
        self.details.rowconfigure(1, weight=1)
        self.details.rowconfigure(2, weight=1)
        ttk.Label(self.details, text="Vyber řádek pro graf konkrétního jádra nebo senzoru.", foreground=MUTED).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4))
        tree_frame = ttk.Frame(self.details)
        tree_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4)
        self.tree = ttk.Treeview(tree_frame, columns=("signal", "value", "source"), show="headings",
                                 selectmode="browse", height=7)
        for key, heading, width in (("signal", "Signál", 245), ("value", "Hodnota", 90), ("source", "Zdroj", 350)):
            self.tree.heading(key, text=heading)
            self.tree.column(key, width=width, stretch=(key != "value"))
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", expand=True, fill="both")
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.select_detail)
        self.detail1 = self.chart(self.details, 2, 0, "Detail signálu", "Výběr v tabulce", "")
        self.detail2 = self.chart(self.details, 2, 1, "Takt jádra", "Dostupný po výběru signálu CPU", "MHz")

        diagnostics = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(diagnostics, text="Diagnostika")
        self.diagnostic = tk.Text(diagnostics, background=PANEL, foreground=TEXT, wrap="word",
                                  relief="flat", padx=11, pady=11, state="disabled")
        self.diagnostic.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.status, foreground=MUTED, wraplength=835, padding=(2, 7, 0, 0)).pack(anchor="w")
        ttk.Label(outer, textvariable=self.app_status, foreground=MUTED, padding=(2, 1, 0, 0)).pack(anchor="w")
        self.render_diagnostics()

    def page(self, title):
        page = ttk.Frame(self.tabs, padding=(0, 5, 0, 0))
        self.tabs.add(page, text=title)
        for index in range(2):
            page.rowconfigure(index, weight=1, uniform="row")
            page.columnconfigure(index, weight=1, uniform="col")
        return page

    def chart(self, parent, row, col, title, subtitle, unit, signals=(), limit=None, margin=4):
        chart = Chart(parent, self, title, subtitle, unit, signals, limit)
        chart.grid(row=row, column=col, sticky="nsew", padx=margin, pady=margin)
        self.charts.append(chart)
        return chart

    def build_dashboard(self):
        page = ttk.Frame(self.tabs, padding=(0, 5, 0, 0))
        self.tabs.add(page, text="Dashboard")
        page.columnconfigure(0, weight=1, uniform="dash")
        page.columnconfigure(1, weight=1, uniform="dash")
        for i in (1, 2):
            page.rowconfigure(i, weight=1, uniform="space")
        controls = ttk.Frame(page, padding=(4, 1, 4, 4))
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(controls, text="Vybrat backendy…", command=self.choose_backends).pack(side="left")
        ttk.Button(controls, text="Vybrat teploty…", command=self.choose_temps).pack(side="left", padx=7)
        self.thermal_status = tk.StringVar(value="Tepelný stav Androidu: —")
        ttk.Label(controls, textvariable=self.thermal_status, foreground=MUTED).pack(side="right")

        frame = ttk.Frame(page)
        frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.backend_tree = ttk.Treeview(frame, columns=("backend", "status", "hint"),
                                         selectmode="browse", show="headings", height=6)
        for key, label, width, should_stretch in (
            ("backend", "Backend", 83, False), ("status", "Aktivita", 175, False),
            ("hint", "Zdroj / poslední údaj", 189, True),
        ):
            self.backend_tree.heading(key, text=label)
            self.backend_tree.column(key, width=width, minwidth=50, stretch=should_stretch)
        self.backend_tree.tag_configure("proof", foreground=MINT)
        self.backend_tree.tag_configure("pending", foreground="#ffc477")
        self.backend_tree.tag_configure("failed", foreground="#ff8496")
        self.backend_tree.tag_configure("indirect", foreground=MUTED)
        self.backend_tree.bind("<<TreeviewSelect>>", self.pick_backend)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.backend_tree.yview)
        self.backend_tree.configure(yscrollcommand=scroll.set)
        self.backend_tree.pack(side="left", expand=True, fill="both")
        scroll.pack(side="right", fill="y")

        self.dash_load = self.chart(page, 1, 1, "Zátěž zařízení a procesu", "CPU procesu jako podíl celkové kapacity CPU", "%",
                                    (("CPU · zátěž", "CPU zařízení"), ("GPU · zátěž", "Adreno"),
                                     ("CPU procesu · zátěž", "CPU aplikace")), 100)
        self.dash_temps = self.chart(page, 2, 0, "Vybrané teploty", "Maxima kategorií nebo konkrétní teplotní zóny", "°C")

        detail = ttk.Frame(page)
        detail.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(1, weight=1)
        self.backend_hint = tk.StringVar(value="Vyber backend v tabulce.")
        ttk.Label(detail, textvariable=self.backend_hint, foreground=MUTED, wraplength=385).grid(
            row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        self.dash_backend = self.chart(detail, 1, 0, "Doba operace", "Vybraný backend · ADBTelemetry", "ms", margin=0)
        self.update_dashboard_filters()

    def choose_backends(self):
        self.choose_signals("Backendy v dashboardu", BACKENDS, self.visible_backends,
                            "Kliknutím vyber nebo odeber backendy.", self.set_visible_backends)

    def choose_temps(self):
        available = tuple(sorted((k for k in self.history if k.startswith("Senzor · ")), key=natural_key))
        self.choose_signals("Teploty v dashboardu", DEFAULT_TEMPS + available, self.visible_temps,
                            "Kliknutím vyber až šest křivek. Během měření se doplní zóny telefonu.",
                            self.set_visible_temperatures, limit=6)

    def choose_signals(self, title, choices, selected, instruction, apply, limit=None):
        dialog = tk.Toplevel(self.root, background=BG)
        dialog.title(title)
        dialog.geometry("535x465")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text=instruction, padding=12, wraplength=490).pack(anchor="w")
        pane = ttk.Frame(dialog, padding=(12, 0, 12, 0))
        pane.pack(fill="both", expand=True)
        listing = tk.Listbox(pane, selectmode="multiple", background=PANEL, foreground=TEXT,
                             selectbackground="#31536b", selectforeground=TEXT, exportselection=False)
        scroll = ttk.Scrollbar(pane, orient="vertical", command=listing.yview)
        listing.configure(yscrollcommand=scroll.set)
        listing.pack(side="left", expand=True, fill="both")
        scroll.pack(side="right", fill="y")
        for i, key in enumerate(choices):
            listing.insert("end", key)
            if key in selected:
                listing.selection_set(i)
        buttons = ttk.Frame(dialog, padding=12)
        buttons.pack(fill="x")

        def save():
            values = [choices[i] for i in listing.curselection()]
            if limit is not None and len(values) > limit:
                messagebox.showwarning(title, f"Vyber nejvýše {limit} křivek.", parent=dialog)
                return
            apply(values)
            dialog.destroy()

        ttk.Button(buttons, text="Použít", style="Accent.TButton", command=save).pack(side="right")
        ttk.Button(buttons, text="Zrušit", command=dialog.destroy).pack(side="right", padx=6)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.bind("<Return>", lambda _event: save())
        listing.focus_set()

    def set_visible_backends(self, values):
        self.visible_backends = values
        self.update_dashboard_filters()

    def set_visible_temperatures(self, values):
        self.visible_temps = values
        self.update_dashboard_filters()

    def update_dashboard_filters(self):
        current_selection = self.backend_tree.selection()
        selected = current_selection[0] if current_selection else None
        self.backend_tree.delete(*self.backend_tree.get_children())
        for backend in self.visible_backends:
            self.backend_tree.insert("", "end", iid=backend)
        if self.visible_backends:
            desired = selected if selected in self.visible_backends else (
                "QNN-HTP" if "QNN-HTP" in self.visible_backends else self.visible_backends[0])
            self.backend_tree.selection_set(desired)
        self.dash_temps.signals = tuple(
            (key, (key.removeprefix("Senzor · ") if key.startswith("Senzor · ") else key.split(" · ")[0])[:18])
            for key in self.visible_temps
        )
        self.update_backend_rows()
        self.paint_all()

    def backend_row(self, backend):
        event = self.backend_status.get(backend)
        library = self.current.get(f"{backend} · knihovna")
        if event:
            state, detail = event
            if state.startswith("Odesláno") and not self.running:
                state, detail = "Měření zastaveno", "Poslední známá událost: odesláno"
            tag = ("proof" if state.startswith("Dokončeno") else "pending" if state.startswith("Odesláno")
                   else "failed" if state in ("Chyba", "Fallback") else "indirect")
            return state, detail, tag
        if backend == "CPU" and "CPU procesu · zátěž" in self.current:
            load = self.current["CPU procesu · zátěž"][0]
            return "Měřeno pro proces", f"/proc/PID/stat · {display(load)} %", "proof"
        if backend in ("CPU", "GPU") and f"{backend} · zátěž" in self.current:
            load, _, source = self.current[f"{backend} · zátěž"]
            return "Celé zařízení", f"{source} · {display(load)} %", "indirect"
        if library and library[0]:
            return "Knihovna načtena", library[2], "indirect"
        if self.timing_enabled:
            return "Bez události", "Čeká na ADBTelemetry", "indirect"
        return "Nedostupné", "Zapni události pro zvolený proces", "indirect"

    def update_backend_rows(self):
        for name in self.visible_backends:
            state, detail, tag = self.backend_row(name)
            self.backend_tree.item(name, values=(name, state, detail), tags=(tag,))
        temp = self.current.get("Android · tepelný stav")
        level = int(temp[0]) if temp is not None else None
        self.thermal_status.set("Tepelný stav Androidu: " + (
            f"{level}/6 · {THERMAL_STATES[level]}" if level is not None else "—"))
        self.pick_backend()

    def pick_backend(self, _event=None):
        selected = self.backend_tree.selection()
        name = selected[0] if selected else ""
        self.dash_backend.title = f"{name} · doba operace" if name else "Doba operace"
        self.dash_backend.signals = ((f"{name} · čas", "Čas"),) if name else ()
        if name:
            state, detail, _ = self.backend_row(name)
            self.backend_hint.set(f"{state} · {detail[:105]}")
        else:
            self.backend_hint.set("Vyber backend v tabulce.")
        self.dash_backend.paint()

    def selected(self):
        index = self.device_box.current()
        return self.devices[index] if 0 <= index < len(self.devices) else ("", "", "")

    def controls(self):
        busy = self.running or self.listing
        self.device_box.configure(state="disabled" if busy or self.demo.get() else "readonly")
        for control in (self.refresh_button, self.browse_button, self.demo_check):
            control.configure(state="disabled" if busy else "normal")
        for control in (self.package_box, self.timing_check):
            control.configure(state="disabled" if self.running else "normal")
        self.period_box.configure(state="disabled" if self.running else "readonly")
        can_start = self.running or (not self.listing and (self.demo.get() or (self.adb and self.selected()[1] == "device")))
        self.start_button.configure(state="normal" if can_start else "disabled",
                                    text="Zastavit" if self.running else "Spustit")
        self.export_button.configure(state="normal" if self.count else "disabled")

    def browse(self):
        path = filedialog.askopenfilename(parent=self.root, title="Vyber adb.exe nebo adb")
        if path:
            self.adb = os.path.abspath(path)
            if not self.demo.get():
                self.refresh()
            else:
                self.render_diagnostics()

    def refresh(self):
        if self.listing or self.running:
            return
        if not self.adb:
            self.adb = find_adb()
        if not self.adb:
            self.status.set("ADB nenalezeno. Zvol Vybrat ADB… a najdi adb.exe v Android SDK Platform-Tools.")
            return
        self.token += 1
        self.listing = True
        self.stop = threading.Event()
        self.status.set("Načítám zařízení přes ADB…")
        self.controls()
        self.launch(refresh_worker, (self.events, self.token, self.stop, self.adb))

    def launch(self, target, args):
        self.workers = [worker for worker in self.workers if worker.is_alive()]
        worker = threading.Thread(target=target, args=args, daemon=True)
        self.workers.append(worker)
        worker.start()

    def toggle_demo(self):
        if self.demo.get():
            self.status.set("Demo je připravené. Klikni na Spustit.")
        elif self.adb:
            self.refresh()
        else:
            self.status.set("Vyber adb.exe ze složky Android SDK Platform-Tools.")
        self.controls()

    def toggle(self):
        if self.running:
            self.halt()
            return
        package = self.package.get().strip()
        if package and not PACKAGE.fullmatch(package):
            messagebox.showwarning("Proces aplikace", "Zadej např. com.example.app nebo com.example.app:worker. Pro sledování hardwaru může být pole prázdné.", parent=self.root)
            return
        demo, timing = self.demo.get(), self.timing.get()
        if not demo and (not self.adb or self.selected()[1] != "device"):
            return
        if self.count and not messagebox.askokcancel(
            "Nová relace", "Nová relace vymaže předchozí grafy a záznam pro export. Pokračovat?", parent=self.root,
        ):
            return
        previous_log = self.log
        try:
            if previous_log:
                previous_log.flush()
            new_log_path = self.log_path.with_name(f"session-{self.token + 1}.csv")
            new_log = new_log_path.open("w", encoding="utf-8-sig", newline="")
        except OSError as exc:
            self.status.set(f"Nelze připravit CSV: {exc}")
            return
        if previous_log:
            previous_log.close()
        self.log, self.log_path = new_log, new_log_path
        self.writer = csv.writer(self.log, delimiter=";")
        self.writer.writerow(("cas_iso", "rezim", "zarizeni", "proces", "skupina", "signal", "hodnota", "jednotka", "zdroj"))
        self.count = 0
        self.history.clear()
        self.known.clear()
        self.current.clear()
        self.notes.clear()
        self.backend_status.clear()
        self.timing_enabled = timing and (bool(package) or demo)
        self.app_status.set("")
        self.running = True
        self.began = time.monotonic()
        self.ended = None
        self.token += 1
        self.stop = threading.Event()
        serial = "DEMO" if demo else self.selected()[0]
        self.session = ("DEMO" if demo else "ADB", serial, package)
        self.connection.set("DEMO · SIMULACE" if demo else "MĚŘENÍ PŘES ADB")
        self.status.set("Čekám na první vzorek…")
        interval = int(self.period.get())
        intervals = [("Hardware", interval), ("Teploty", max(2, interval))]
        if package or demo:
            intervals.append(("Aplikace", max(3, interval)))
        if self.timing_enabled:
            intervals.append(("Události", .75))
        if not package and not demo:
            self.app_status.set("Sledování procesu je vypnuté. Pole Proces aplikace je prázdné.")
        for group, delay in intervals:
            args = (self.events, self.token, self.stop, self.adb, serial, package, group, delay, demo)
            self.launch(sample_worker, args)
        self.refresh_views()
        self.controls()

    def halt(self):
        self.stop.set()
        self.running = False
        self.ended = time.monotonic()
        self.token += 1
        self.connection.set("DEMO · ZASTAVENO" if self.session[0] == "DEMO" else "ZASTAVENO")
        self.status.set("Měření zastaveno. Grafy a export zůstávají dostupné do začátku nové relace.")
        self.update_backend_rows()
        self.controls()

    def drain(self):
        while True:
            try:
                token, mode, group, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if token != self.token:
                continue
            if mode in ("devices", "device_error"):
                self.finish_listing(mode, payload)
            elif mode == "sample":
                timewall, data, note, statuses = payload
                if group == "Události":
                    self.backend_status = statuses or {}
                self.notes[group] = f"{timewall:%H:%M:%S} · {len(data)} signálů" + (f" · {note}" if note else "")
                if group == "Aplikace":
                    self.app_status.set(note)
                self.record(group, timewall, data)
                self.show_status()
            elif mode == "error":
                if group == "Události":
                    self.backend_status = {}
                self.record(group, datetime.now().astimezone(), {})
                self.notes[group] = payload
                if group == "Aplikace":
                    self.app_status.set("Vzorek procesu nedostupný.")
                self.show_status()
        self.drain_id = self.root.after(120, self.drain)

    def finish_listing(self, mode, payload):
        self.listing = False
        previous = self.selected()[0]
        self.devices = payload if mode == "devices" else []
        self.device_box.configure(values=[d[2] for d in self.devices])
        if self.devices:
            fallback = next((i for i, d in enumerate(self.devices) if d[1] == "device"), 0)
            self.device_box.current(next((i for i, d in enumerate(self.devices) if d[0] == previous), fallback))
        else:
            self.device.set("")
        if mode == "device_error":
            self.status.set(f"Výpis zařízení selhal. {payload}")
        elif not self.devices:
            self.status.set("ADB nevidí žádné zařízení. Připoj telefon, povol ladění USB a obnov seznam.")
        elif all(d[1] != "device" for d in self.devices):
            self.status.set("Žádné zařízení není připravené. Odemkni telefon a potvrď ladění USB nebo obnov připojení.")
        else:
            self.status.set("Vyber zařízení a klikni na Spustit." if not self.demo.get() else "Demo je připravené.")
        self.controls()
        self.render_diagnostics()

    def record(self, group, stamp, data):
        known = self.known.setdefault(group, set())
        known.update(data)
        recorded = time.monotonic()
        for key in known:
            item = data.get(key)
            points = self.history.setdefault(key, deque())
            points.append((recorded, item[0] if item else None))
            while points and points[0][0] < recorded - HISTORY:
                points.popleft()
            if item:
                self.current[key] = item
            else:
                self.current.pop(key, None)
        try:
            if data and self.log:
                prefix = (stamp.isoformat(timespec="milliseconds"), *self.session, group)
                for key, (value, unit, source) in sorted(data.items()):
                    row = (*prefix, key, f"{value:.6f}".rstrip("0").rstrip(".").replace(".", ","), unit, source)
                    self.writer.writerow(csv_safe(cell) for cell in row)
                self.log.flush()
                self.count += len(data)
        except OSError as exc:
            self.app_status.set(f"Zápis CSV selhal: {exc}")
            self.halt()
        self.refresh_views()
        self.controls()

    def show_status(self):
        if self.running:
            self.status.set("   |   ".join(f"{group}: {self.notes[group].partition(chr(10))[0][:82]}" for group in self.notes))
        self.render_diagnostics()

    def render_diagnostics(self):
        signals = (
            ("CPU zařízení", self.current.get("CPU · zátěž")),
            ("CPU procesu", self.current.get("CPU procesu · zátěž")),
            ("Adreno zátěž", self.current.get("GPU · zátěž")),
            ("Adreno takt", self.current.get("GPU · takt")),
            ("NPU teplota", self.current.get("NPU · teplota")),
            ("CDSP teplota", self.current.get("CDSP · teplota")),
        )
        sections_text = [
            f"ADB: {self.adb or 'nenalezeno'}",
            "Zařízení: " + ("; ".join(item[2] for item in self.devices) or "—"),
            *[f"{k}: {v}" for k, v in self.notes.items()],
            "",
            *[f"{name}: {display(info[0])} {info[1]} · {info[2]}" if info else
              f"{name}: nedostupné v aktuálním vzorku" for name, info in signals],
            "Teplotní zóny: " + str(sum(k.startswith("Senzor · ") for k in self.current)),
            "Čitelné domény devfreq: " + (", ".join(k.split(" · ", 1)[1] for k in self.current
                                                  if k.startswith("Devfreq · ")) or "—"),
            "",
            "Chybějící měření se zobrazí jako pomlčka a mezera v grafu.",
            "Názvy teplotních zón určuje výrobce. Karta Jádra a senzory ukazuje originální názvy.",
            "Takt CDSP/devfreq je dostupný jen u viditelných uzlů. Nevyjadřuje procento vytížení NPU.",
            "Mapy procesu se čtou přes run-as. Přítomnost knihovny může souviset také s vykreslováním UI.",
            "Dashboard používá značky ADBTelemetry. Zahájení, dokončení, fallback i knihovna mají vlastní stav.",
            "Stav Androidu je celosystémová úroveň tepelného omezení od 0 do 6.",
            "Veškeré signály se sbírají i po skrytí z dashboardu.",
        ]
        self.diagnostic.configure(state="normal")
        self.diagnostic.delete("1.0", "end")
        self.diagnostic.insert("1.0", "\n".join(sections_text))
        self.diagnostic.configure(state="disabled")

    def refresh_views(self):
        visible_freq = sorted(k for k in self.current if k.startswith("Devfreq · "))
        old_freq = [k for k, _ in self.freq_chart.signals if k.startswith("Devfreq · ") and k in self.history]
        ordered_freq = list(dict.fromkeys(visible_freq + old_freq))
        self.freq_chart.signals = (("GPU · takt", "GPU"),) + tuple(
            (key, key.split(" · ", 1)[1][:21]) for key in ordered_freq[:3]
        )
        self.update_backend_rows()
        all_details = {
            k for k in self.history
            if k.startswith(("CPU ", "Senzor · ", "Devfreq · "))
        }
        for key in sorted(all_details - self.details_by_row.keys(), key=natural_key):
            index = sum(natural_key(other) < natural_key(key) for other in self.details_by_row)
            self.details_by_row[key] = self.tree.insert("", index, values=(key, "—", "—"))
        for key in list(self.details_by_row):
            row = self.details_by_row[key]
            if key not in all_details:
                self.tree.delete(row)
                del self.details_by_row[key]
                continue
            info = self.current.get(key)
            self.tree.item(row, values=(key, f"{display(info[0])} {info[1]}" if info else "—", info[2] if info else "—"))
        self.select_detail()
        self.paint_all()

    def select_detail(self, _event=None):
        chosen = self.tree.selection()
        signal = self.tree.item(chosen[0], "values")[0] if chosen else ""
        core = re.match(r"(CPU \d+) · ", signal)
        if core:
            primary, secondary = f"{core[1]} · zátěž", f"{core[1]} · takt"
            self.detail1.title, self.detail1.unit, self.detail1.limit = primary, "%", 100
            self.detail1.signals = ((primary, "Zátěž"),)
            self.detail2.title, self.detail2.signals = secondary, ((secondary, "Takt"),)
        elif signal:
            self.detail1.title = signal[:64]
            self.detail1.unit = "°C" if signal.startswith("Senzor") else "MHz"
            self.detail1.limit, self.detail1.signals = None, ((signal, "Měření"),)
            self.detail2.title, self.detail2.signals = "Takt jádra", ()
        else:
            self.detail1.title, self.detail1.unit, self.detail1.limit = "Detail signálu", "", None
            self.detail1.signals = ()
            self.detail2.title, self.detail2.signals = "Takt jádra", ()
        self.detail1.paint()
        self.detail2.paint()

    def paint_all(self):
        for chart in self.charts:
            if chart.winfo_viewable():
                chart.paint()

    def redraw(self):
        self.paint_all()
        self.redraw_id = self.root.after(600, self.redraw)

    def export(self):
        if not self.count:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Uložit měření",
                                           defaultextension=".csv", filetypes=(("CSV", "*.csv"),),
                                           initialfile=f"adb-telemetrie-{datetime.now():%Y%m%d-%H%M%S}.csv")
        if not path:
            return
        pending = None
        try:
            self.log.flush()
            with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as out:
                pending = Path(out.name)
                with self.log_path.open("rb") as source:
                    shutil.copyfileobj(source, out)
            os.replace(pending, path)
        except OSError as exc:
            messagebox.showerror("Export CSV", str(exc), parent=self.root)
        else:
            messagebox.showinfo("Export CSV", f"Měření uloženo do\n{path}", parent=self.root)
        finally:
            if pending:
                pending.unlink(missing_ok=True)

    def close(self):
        self.stop.set()
        self.root.after_cancel(self.drain_id)
        self.root.after_cancel(self.redraw_id)
        deadline = time.monotonic() + .5
        for worker in self.workers:
            worker.join(max(0, deadline - time.monotonic()))
        try:
            if self.log:
                self.log.close()
            self.temp_dir.cleanup()
        finally:
            self.root.destroy()


def natural_key(name):
    return [int(token) if token.isdigit() else token.casefold() for token in re.split(r"(\d+)", name)]


def csv_safe(text):
    # Preserve numeric values while preventing device-provided strings from becoming spreadsheet formulas.
    if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")) and not re.fullmatch(r"-?\d+(?:,\d+)?", text):
        return "'" + text
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="start with demo mode selected")
    parser.add_argument("--adb", help="absolute path to adb or adb.exe")
    args = parser.parse_args()
    root = tk.Tk()
    Monitor(root, args.demo, args.adb)
    root.mainloop()


if __name__ == "__main__":
    main()
