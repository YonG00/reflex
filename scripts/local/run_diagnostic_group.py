"""Bound a diagnostic's process group; kill descendants even if shell exits."""

import argparse
import os
import signal
import subprocess
import sys
import time


def alive(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def stop_group(process, grace):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    end = time.monotonic() + grace
    while time.monotonic() < end and alive(process.pid):
        process.poll()
        time.sleep(0.05)
    if alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run(command, timeout, grace):
    # Reset inherited nohup signal dispositions in a fresh child interpreter.
    launcher = "import os,signal,sys;signal.signal(signal.SIGINT,signal.SIG_DFL);signal.signal(signal.SIGTERM,signal.SIG_DFL);os.execvp(sys.argv[1],sys.argv[1:])"
    process = subprocess.Popen(
        [sys.executable, "-c", launcher, *command], start_new_session=True
    )

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_group(process, grace)
            return 124
        except KeyboardInterrupt:
            stop_group(process, grace)
            return 130
        # A parent may exit while a child still owns CUDA. Clean the private group.
        if alive(process.pid):
            stop_group(process, grace)
        return code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            stop_group(process, grace)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--grace", type=float, default=5)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or a.timeout <= 0 or a.grace < 0:
        p.error("Positive timeout, nonnegative grace and command required")
    raise SystemExit(run(command, a.timeout, a.grace))


if __name__ == "__main__":
    main()
