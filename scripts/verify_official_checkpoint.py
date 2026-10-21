#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("config", type=Path)
args = parser.parse_args()

checkpoint = args.checkpoint.resolve()
required = [checkpoint / name for name in ("agent.th", "mixer.th", "opt.th", "resume.th")]
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"invalid checkpoint file: {path}")
if checkpoint.name != "30000":
    raise SystemExit(f"expected final 30000 checkpoint, got {checkpoint}")

config = json.loads(args.config.read_text())
expected = {
    "agent": "mt_hissd",
    "learner": "hissd_learner",
    "optim_type": "AdamW",
    "lr": 0.0003,
    "critic_lr": 0.0001,
    "weight_decay": 0.0001,
    "beta": 0.05,
    "td_weight": 10.0,
    "coef_kl": 0.1,
    "ssl_type": "moco",
    "mixer": "mt_qattn",
    "seed": 2,
    "t_max": 30000,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"config mismatch {key}: {config.get(key)!r} != {value!r}")
if config.get("train_tasks") != ["2s3z", "2s4z", "3s5z"]:
    raise SystemExit(f"unexpected train_tasks: {config.get('train_tasks')}")

agent = torch.load(required[0], map_location="cpu")
adapter_keys = [key for key in agent if "adapter" in key or "team_encoder" in key]
if adapter_keys:
    raise SystemExit(f"official checkpoint contains adapter keys: {adapter_keys[:5]}")
optimizer = torch.load(checkpoint / "opt.th", map_location="cpu")
group = optimizer["param_groups"][0]
if group["lr"] != 0.0001 or group["weight_decay"] != 0.0001:
    raise SystemExit(f"optimizer mismatch: {group}")
resume = torch.load(checkpoint / "resume.th", map_location="cpu", weights_only=False)
if int(resume.get("t_env", -1)) < 30000 or int(resume.get("version", -1)) != 3:
    raise SystemExit("resume state is not a completed 30k V3 checkpoint")

for path in required:
    print(f"{sha256(path)}  {path.name}")
