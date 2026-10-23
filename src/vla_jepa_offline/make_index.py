"""Generate JSONL from task_XX/traj*.pkl, with explicit user-supplied task prompts.

Task prompts JSON e.g. {"00":"Pick the green cube and place it into bin 1",...}
The same script is used for E3 and E4. No implicit data split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", required=True, type=Path)
    cli.add_argument("--task-prompts", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--exclude-task-ids", default="0,5,10,15")
    cli.add_argument("--prefix", default="offline")
    args = cli.parse_args()
    root = args.dataset_root.expanduser().resolve()
    prompts = json.loads(args.task_prompts.read_text("utf-8"))
    excluded = {int(x.strip()) for x in args.exclude_task_ids.split(",") if x.strip()}
    entries = []
    for task_dir in sorted(root.glob("task_*")):
        if not task_dir.is_dir():
            continue
        tid = int(task_dir.name.removeprefix("task_"))
        if tid in excluded:
            continue
        key = f"{tid:02d}"
        if key not in prompts or not isinstance(prompts[key], str) or not prompts[key].strip():
            raise ValueError(f"Missing task prompt {key} in {args.task_prompts}")
        for p in sorted(task_dir.glob("traj*.pkl")):
            entries.append({
                "episode_id": f"{args.prefix}_{task_dir.name}_{p.stem}",
                "task_id": tid,
                "task": prompts[key],
                "pkl_path": str(p.resolve()),
            })
    if not entries:
        raise ValueError(f"No PKL episodes found under {root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable index: {args.output}")
    with args.output.open("x", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Created {args.output} with {len(entries)} episodes; held-out={sorted(excluded)}")


if __name__ == "__main__":
    main()
