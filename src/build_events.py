#!/usr/bin/env python3
"""Build time-ordered recommendation events for the RL task.

Each event represents one worker arrival. The historical project submitted by
the worker is used as the positive project, and K-1 active projects at the same
time are sampled as negative candidates.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


EVENT_FIELDS = [
    "event_id",
    "split",
    "event_time",
    "worker_id",
    "positive_project_id",
    "candidate_project_ids",
    "label_index",
    "active_project_count",
    "entry_id",
    "entry_number",
    "winner",
    "finalist",
    "withdrawn",
    "award_value",
    "offer_value",
    "tip_value",
    "score",
    "normalized_score",
    "is_entirely_original",
    "worker_quality",
    "quality_unknown",
]


@dataclass(frozen=True)
class ProjectWindow:
    project_id: int
    start: datetime
    deadline: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RL recommendation events from processed CSV files."
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("outputs/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/processed"))
    parser.add_argument("--candidate_size", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_fallback_negatives",
        action="store_true",
        help=(
            "If active projects are insufficient, fill negatives from all projects. "
            "Default is to skip such events."
        ),
    )
    return parser.parse_args()


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_projects(project_path: Path) -> tuple[list[ProjectWindow], set[int]]:
    windows: list[ProjectWindow] = []
    project_ids: set[int] = set()
    for row in read_csv(project_path):
        project_id = parse_int(row.get("project_id"), default=-1)
        start = parse_time(row.get("start_date", ""))
        deadline = parse_time(row.get("deadline", ""))
        if project_id < 0:
            continue
        project_ids.add(project_id)
        if start is None or deadline is None or deadline < start:
            continue
        windows.append(ProjectWindow(project_id=project_id, start=start, deadline=deadline))
    windows.sort(key=lambda item: (item.start, item.project_id))
    return windows, project_ids


def load_workers(worker_path: Path) -> dict[int, dict[str, str]]:
    workers: dict[int, dict[str, str]] = {}
    for row in read_csv(worker_path):
        worker_id = parse_int(row.get("worker_id"), default=-1)
        if worker_id >= 0:
            workers[worker_id] = row
    return workers


def load_entries(entry_path: Path, valid_project_ids: set[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(entry_path):
        project_id = parse_int(row.get("project_id"), default=-1)
        event_time = parse_time(row.get("entry_created_at", ""))
        worker_id = parse_int(row.get("worker_id"), default=-1)
        if project_id not in valid_project_ids or event_time is None or worker_id < 0:
            continue
        row["_event_dt"] = event_time.isoformat()
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["_event_dt"],
            parse_int(row.get("project_id"), default=-1),
            parse_int(row.get("entry_number"), default=0),
            parse_int(row.get("entry_id"), default=0),
        )
    )
    return rows


class ActiveProjectSampler:
    """Maintains active projects while scanning events by time."""

    def __init__(self, projects: list[ProjectWindow]) -> None:
        self.projects = projects
        self.next_project_idx = 0
        self.active: set[int] = set()
        self.deadline_heap: list[tuple[datetime, int]] = []
        self.cached_tuple: tuple[int, ...] = ()
        self.cache_dirty = True

    def update(self, now: datetime) -> None:
        changed = False
        while (
            self.next_project_idx < len(self.projects)
            and self.projects[self.next_project_idx].start <= now
        ):
            project = self.projects[self.next_project_idx]
            self.active.add(project.project_id)
            heapq.heappush(self.deadline_heap, (project.deadline, project.project_id))
            self.next_project_idx += 1
            changed = True

        while self.deadline_heap and self.deadline_heap[0][0] < now:
            _, project_id = heapq.heappop(self.deadline_heap)
            if project_id in self.active:
                self.active.remove(project_id)
                changed = True

        if changed:
            self.cache_dirty = True

    def active_tuple(self) -> tuple[int, ...]:
        if self.cache_dirty:
            self.cached_tuple = tuple(sorted(self.active))
            self.cache_dirty = False
        return self.cached_tuple


def choose_split(index: int, total: int, train_ratio: float, valid_ratio: float) -> str:
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    if index < train_end:
        return "train"
    if index < valid_end:
        return "valid"
    return "test"


def sample_candidates(
    *,
    rng: random.Random,
    positive_project_id: int,
    active_project_ids: tuple[int, ...],
    all_project_ids: tuple[int, ...],
    candidate_size: int,
    allow_fallback_negatives: bool,
) -> tuple[list[int] | None, int]:
    active_negatives = [pid for pid in active_project_ids if pid != positive_project_id]
    needed = candidate_size - 1

    if len(active_negatives) >= needed:
        negatives = rng.sample(active_negatives, needed)
    elif allow_fallback_negatives:
        negatives = list(active_negatives)
        fallback_pool = [
            pid for pid in all_project_ids if pid != positive_project_id and pid not in set(negatives)
        ]
        if len(fallback_pool) < needed - len(negatives):
            return None, len(active_project_ids)
        negatives.extend(rng.sample(fallback_pool, needed - len(negatives)))
    else:
        return None, len(active_project_ids)

    candidates = negatives + [positive_project_id]
    rng.shuffle(candidates)
    return candidates, len(active_project_ids)


def build_event_row(
    *,
    event_id: int,
    split: str,
    entry: dict[str, str],
    candidates: list[int],
    active_project_count: int,
    workers: dict[int, dict[str, str]],
) -> dict[str, Any]:
    positive_project_id = parse_int(entry.get("project_id"), default=-1)
    worker_id = parse_int(entry.get("worker_id"), default=-1)
    worker = workers.get(worker_id, {})

    return {
        "event_id": event_id,
        "split": split,
        "event_time": entry.get("entry_created_at", ""),
        "worker_id": worker_id,
        "positive_project_id": positive_project_id,
        "candidate_project_ids": "|".join(str(pid) for pid in candidates),
        "label_index": candidates.index(positive_project_id),
        "active_project_count": active_project_count,
        "entry_id": parse_int(entry.get("entry_id")),
        "entry_number": parse_int(entry.get("entry_number")),
        "winner": parse_int(entry.get("winner")),
        "finalist": parse_int(entry.get("finalist")),
        "withdrawn": parse_int(entry.get("withdrawn")),
        "award_value": parse_float(entry.get("award_value")),
        "offer_value": parse_float(entry.get("offer_value")),
        "tip_value": parse_float(entry.get("tip_value")),
        "score": parse_float(entry.get("score")),
        "normalized_score": parse_float(entry.get("normalized_score")),
        "is_entirely_original": parse_int(entry.get("is_entirely_original")),
        "worker_quality": parse_float(worker.get("worker_quality")),
        "quality_unknown": parse_int(worker.get("quality_unknown"), default=1),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.candidate_size < 2:
        raise ValueError("--candidate_size must be at least 2")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")
    if not 0 <= args.valid_ratio < 1:
        raise ValueError("--valid_ratio must be between 0 and 1")
    if args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("--train_ratio + --valid_ratio must be smaller than 1")

    processed_dir = args.processed_dir
    output_dir = args.output_dir
    project_path = processed_dir / "project.csv"
    entry_path = processed_dir / "entry.csv"
    worker_path = processed_dir / "worker.csv"

    projects, project_ids = load_projects(project_path)
    workers = load_workers(worker_path)
    entries = load_entries(entry_path, project_ids)
    sampler = ActiveProjectSampler(projects)
    all_project_ids = tuple(sorted(project_ids))
    rng = random.Random(args.seed)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    skipped_insufficient_negatives = 0
    skipped_time_parse = 0

    for source_idx, entry in enumerate(entries):
        event_dt = parse_time(entry.get("entry_created_at", ""))
        if event_dt is None:
            skipped_time_parse += 1
            continue
        sampler.update(event_dt)
        positive_project_id = parse_int(entry.get("project_id"), default=-1)
        candidates, active_count = sample_candidates(
            rng=rng,
            positive_project_id=positive_project_id,
            active_project_ids=sampler.active_tuple(),
            all_project_ids=all_project_ids,
            candidate_size=args.candidate_size,
            allow_fallback_negatives=args.allow_fallback_negatives,
        )
        if candidates is None:
            skipped_insufficient_negatives += 1
            continue

        split = choose_split(source_idx, len(entries), args.train_ratio, args.valid_ratio)
        event_id = sum(len(rows) for rows in split_rows.values())
        row = build_event_row(
            event_id=event_id,
            split=split,
            entry=entry,
            candidates=candidates,
            active_project_count=active_count,
            workers=workers,
        )
        split_rows[split].append(row)

    write_csv(output_dir / "events_train.csv", split_rows["train"])
    write_csv(output_dir / "events_valid.csv", split_rows["valid"])
    write_csv(output_dir / "events_test.csv", split_rows["test"])

    stats = {
        "candidate_size": args.candidate_size,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.valid_ratio,
        "project_count": len(project_ids),
        "project_with_valid_window_count": len(projects),
        "entry_count_after_basic_filter": len(entries),
        "event_count": sum(len(rows) for rows in split_rows.values()),
        "train_event_count": len(split_rows["train"]),
        "valid_event_count": len(split_rows["valid"]),
        "test_event_count": len(split_rows["test"]),
        "skipped_insufficient_negatives": skipped_insufficient_negatives,
        "skipped_time_parse": skipped_time_parse,
        "allow_fallback_negatives": args.allow_fallback_negatives,
    }
    write_json(output_dir / "events_stats.json", stats)

    print(f"Wrote {len(split_rows['train'])} train events to {output_dir / 'events_train.csv'}")
    print(f"Wrote {len(split_rows['valid'])} valid events to {output_dir / 'events_valid.csv'}")
    print(f"Wrote {len(split_rows['test'])} test events to {output_dir / 'events_test.csv'}")
    print(f"Wrote event stats to {output_dir / 'events_stats.json'}")
    if skipped_insufficient_negatives:
        print(
            "Warning: skipped "
            f"{skipped_insufficient_negatives} events because active negatives were insufficient."
        )


if __name__ == "__main__":
    main()

