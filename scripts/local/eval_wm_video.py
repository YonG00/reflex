"""Evaluate one explicitly selected WM checkpoint; no training or model writes."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def command_for(checkpoint, config, output, seconds, dt, seed, num_envs):
    steps = max(2, round(seconds / dt) + 1)
    return [
        "bash",
        str(ROOT / "scripts/train_stage1_world_model.sh"),
        "--mode",
        "play",
        "--video",
        "--no_debug_vis",
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--play_steps",
        str(steps),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--reinitialize_replay",
        "--migrate_event_execution",
        "--output",
        str(output),
    ]


def write_flags(flags, path):
    rows = []
    for key, value in sorted(flags.items()):
        if not re.fullmatch(r"(?:BOX_|AMP_W_|M5_)[A-Za-z0-9_]+", key):
            raise ValueError(f"Unsupported checkpoint environment key: {key}")
        rows.append(f"{key}={shlex.quote(str(value))}")
    path.write_text("\n".join(rows) + "\n")


def validate_video(path):
    binary = shutil.which("ffmpeg")
    if binary is None:
        import imageio_ffmpeg

        binary = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            binary,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-progress",
            "pipe:1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    progress = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    if int(progress.get("frame", "0")) < 1:
        raise RuntimeError("Video has no decodable frames")
    checked = {
        "full_decode_passed": True,
        "frames": int(progress["frame"]),
        "duration_seconds": int(progress["out_time_us"]) / 1000000,
    }
    probe = shutil.which("ffprobe")
    if probe:
        data = json.loads(
            subprocess.check_output(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )
        checked["duration_seconds"] = float(data["format"]["duration"])
        checked.update(data["streams"][0])
    return checked


def main():
    parser = argparse.ArgumentParser(
        description="指定WM联合训练checkpoint，自动匹配配置并录制无坐标轴视频。"
    )
    parser.add_argument(
        "checkpoint", type=Path, help="模型.pt路径，可用绝对路径或仓库内相对路径"
    )
    parser.add_argument("--seconds", type=float, default=15, help="视频秒数，默认15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-envs", type=int, default=1, help="默认单环境，画面无其他机器人"
    )
    parser.add_argument(
        "--output", type=Path, help="空输出目录；默认按时间和模型名自动创建"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="仅导出配置和命令，不启动仿真"
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.num_envs < 1:
        parser.error("seconds须为有限正数，num-envs须>=1")
    checkpoint = args.checkpoint.expanduser()
    checkpoint = (
        (ROOT / checkpoint).resolve()
        if not checkpoint.is_absolute()
        else checkpoint.resolve()
    )
    if not checkpoint.is_file():
        parser.error(f"模型不存在: {checkpoint}")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("config"), dict)
        or "actor" not in state
        or "model" not in state
    ):
        parser.error(
            "需要WM联合训练checkpoint；原Reflex PPO的model_*.pt请使用PPO评测入口"
        )
    config = dict(state["config"])
    dt = (state.get("task_spec") or {}).get("dt", 0.02)
    if not isinstance(dt, (float, int)) or dt <= 0:
        parser.error("checkpoint控制步长无效")
    source = {
        k: state.get(k)
        for k in ("iteration", "actor_updates", "model_updates", "termination_updates")
    }
    flags = state.get("contract", {}).get("flags", {})
    del state
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = (
        args.output or Path("outputs/world_model/videos") / f"{stamp}_{checkpoint.stem}"
    )
    output = output.expanduser()
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"输出目录非空，请换新目录: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path, flags_path = (
        output / "checkpoint_config.json",
        output / "checkpoint_flags.env",
    )
    config_path.write_text(json.dumps(config, indent=2))
    write_flags(flags, flags_path)
    command = command_for(
        checkpoint, config_path, output, args.seconds, dt, args.seed, args.num_envs
    )
    request = {
        "checkpoint": str(checkpoint),
        "source": source,
        "seconds_requested": args.seconds,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "debug_visualization": False,
        "inference_only": True,
        "command": command,
    }
    (output / "evaluation_request.json").write_text(json.dumps(request, indent=2))
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + f"cd {shlex.quote(str(ROOT))}\nsource scripts/local/env.sh\nexport REFLEX_WM_ENV_FILE={shlex.quote(str(flags_path))}\n"
        + shlex.join(command)
        + "\n"
    )
    (output / "command.sh").write_text(script)
    print(
        f"模型: {checkpoint}\n配置: {config.get('reward_profile', 'original')}, Actor H{config.get('horizon')}\n输出: {output}\n命令: {shlex.join(command)}",
        flush=True,
    )
    if args.dry_run:
        print(f"仅生成命令；可执行: bash {shlex.quote(str(output / 'command.sh'))}")
        return
    env = dict(os.environ, REFLEX_WM_ENV_FILE=str(flags_path))
    with (output / "run.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    if code:
        raise RuntimeError(f"评测失败，退出码{code}，查看 {output / 'run.log'}")
    summary_path = output / "play_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("评测没有完成结果，不能称为成功")
    summary = json.loads(summary_path.read_text())
    if (
        source["actor_updates"] is not None
        and summary["actor_updates_in_checkpoint"] != source["actor_updates"]
    ):
        raise RuntimeError("加载的模型更新计数与请求不符")
    videos = sorted((output / "videos").glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"预期一个视频，实际{len(videos)}个")
    checked = validate_video(videos[0])
    video = output / "model_video.mp4"
    shutil.copy2(videos[0], video)
    (output / "video_validation.json").write_text(json.dumps(checked, indent=2))
    print(
        f"\n评测完成：{checked['duration_seconds']:.2f}秒，{checked['frames']}帧，坐标轴已隐藏。\n视频: {video}\n指标: {summary_path}\n日志: {output / 'run.log'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
