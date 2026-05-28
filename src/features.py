#!/usr/bin/env python3
"""Build numeric feature tensors for DQN training.

Inputs:
  outputs/processed/project.csv
  outputs/processed/events_train.csv
  outputs/processed/events_valid.csv
  outputs/processed/events_test.csv

Outputs:
  outputs/features/train_features.npz
  outputs/features/valid_features.npz
  outputs/features/test_features.npz
  outputs/features/feature_names.json
  outputs/features/features_stats.json

The script scans events in chronological order and updates worker/project
history only after the current event is featurized. This avoids using future
information when computing dynamic preference and count features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
    raise SystemExit(
        "Missing dependency: numpy. Please run `conda env create -f environment.yml` "
        "and then `conda activate rl-course` before running src/features.py."
    ) from exc


SPLITS = ("train", "valid", "test")

FEATURE_NAMES = [
    "bias",
    "worker_quality",
    "worker_quality_unknown",
    "worker_submit_count_log_norm",
    "worker_avg_score",
    "worker_win_rate",
    "worker_finalist_rate",
    "worker_withdrawn_rate",
    "worker_avg_award_log_norm",
    "project_category_norm",
    "project_sub_category_norm",
    "project_industry_norm",
    "project_award_log_norm",
    "project_tips_log_norm",
    "project_expected_answer_log_norm",
    "project_total_entry_count_log_norm",
    "project_creative_count_log_norm",
    "project_average_score_norm",
    "project_featured",
    "project_assured",
    "project_private_gallery",
    "project_duration_log_norm",
    "project_age_ratio",
    "project_time_left_ratio",
    "project_current_entries_log_norm",
    "project_current_avg_score",
    "project_current_winner_rate",
    "project_current_finalist_rate",
    "project_current_withdrawn_rate",
    "category_match",
    "industry_match",
    "worker_category_preference",
    "worker_industry_preference",
    "worker_project_history_log_norm",
    "active_project_count_log_norm",
]


@dataclass
class ProjectInfo:
    project_id: int
    category: int
    sub_category: int
    industry_id: int
    start_date: datetime | None
    deadline: datetime | None
    duration_hours: float
    award_log: float
    tips_log: float
    expected_answer_num: float
    entry_count: float
    creative_count: float
    average_score: float
    featured: float
    assured: float
    private_gallery: float


@dataclass
class WorkerHistory:
    submit_count: int = 0
    score_sum: float = 0.0
    winner_count: int = 0
    finalist_count: int = 0
    withdrawn_count: int = 0
    award_log_sum: float = 0.0
    category_counts: Counter[int] = field(default_factory=Counter)
    industry_counts: Counter[int] = field(default_factory=Counter)
    project_counts: Counter[int] = field(default_factory=Counter)


@dataclass
class ProjectHistory:
    submit_count: int = 0
    score_sum: float = 0.0
    winner_count: int = 0
    finalist_count: int = 0
    withdrawn_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert recommendation events into numeric feature tensors."
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("outputs/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/features"))
    parser.add_argument(
        "--max_events_per_split",
        type=int,
        default=0,
        help="Debug option. If > 0, keep at most this many events per split.",
    )
    return parser.parse_args()


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
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
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def log_norm(value: float, max_log_value: float) -> float:
    if max_log_value <= 0:
        return 0.0
    return math.log1p(max(value, 0.0)) / max_log_value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_projects(path: Path) -> dict[int, ProjectInfo]:
    projects: dict[int, ProjectInfo] = {}
    for row in read_csv(path):
        project_id = parse_int(row.get("project_id"), default=-1)
        if project_id < 0:
            continue
        projects[project_id] = ProjectInfo(
            project_id=project_id,
            category=parse_int(row.get("category")),
            sub_category=parse_int(row.get("sub_category")),
            industry_id=parse_int(row.get("industry_id")),
            start_date=parse_time(row.get("start_date", "")),
            deadline=parse_time(row.get("deadline", "")),
            duration_hours=parse_float(row.get("duration_hours")),
            award_log=parse_float(row.get("award_log")),
            tips_log=parse_float(row.get("tips_log")),
            expected_answer_num=parse_float(row.get("project_list_answer_num")),
            entry_count=parse_float(row.get("entry_count")),
            creative_count=parse_float(row.get("creative_count")),
            average_score=parse_float(row.get("average_score")),
            featured=parse_float(row.get("featured")),
            assured=parse_float(row.get("assured")),
            private_gallery=parse_float(row.get("private_gallery")),
        )
    return projects


def load_events(processed_dir: Path) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for split in SPLITS:
        for row in read_csv(processed_dir / f"events_{split}.csv"):
            row["_split"] = split
            row["_event_dt"] = row.get("event_time", "")
            events.append(row)
    events.sort(key=lambda row: (row["_event_dt"], parse_int(row.get("event_id"))))
    return events


def compute_normalizers(projects: dict[int, ProjectInfo], events: list[dict[str, str]]) -> dict[str, float]:
    max_category = max((p.category for p in projects.values()), default=1)
    max_sub_category = max((p.sub_category for p in projects.values()), default=1)
    max_industry = max((p.industry_id for p in projects.values()), default=1)
    max_award_log = max((p.award_log for p in projects.values()), default=1.0)
    max_tips_log = max((p.tips_log for p in projects.values()), default=1.0)
    max_expected_answer_log = max(
        (math.log1p(p.expected_answer_num) for p in projects.values()), default=1.0
    )
    max_entry_count_log = max((math.log1p(p.entry_count) for p in projects.values()), default=1.0)
    max_creative_count_log = max(
        (math.log1p(p.creative_count) for p in projects.values()), default=1.0
    )
    max_duration_log = max((math.log1p(p.duration_hours) for p in projects.values()), default=1.0)
    max_active_log = max(
        (math.log1p(parse_float(row.get("active_project_count"))) for row in events), default=1.0
    )

    return {
        "max_category": max(max_category, 1),
        "max_sub_category": max(max_sub_category, 1),
        "max_industry": max(max_industry, 1),
        "max_award_log": max(max_award_log, 1e-6),
        "max_tips_log": max(max_tips_log, 1e-6),
        "max_expected_answer_log": max(max_expected_answer_log, 1e-6),
        "max_entry_count_log": max(max_entry_count_log, 1e-6),
        "max_creative_count_log": max(max_creative_count_log, 1e-6),
        "max_duration_log": max(max_duration_log, 1e-6),
        "max_active_log": max(max_active_log, 1e-6),
        "max_worker_submit_log": math.log1p(len(events)),
    }


def allocate_outputs(
    events: list[dict[str, str]],
    candidate_size: int,
    feature_dim: int,
    max_events_per_split: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    counts = {split: 0 for split in SPLITS}
    for row in events:
        split = row["_split"]
        if max_events_per_split and counts[split] >= max_events_per_split:
            continue
        counts[split] += 1

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for split, count in counts.items():
        outputs[split] = {
            "states": np.zeros((count, candidate_size, feature_dim), dtype=np.float32),
            "labels": np.zeros((count,), dtype=np.int64),
            "worker_rewards": np.zeros((count, candidate_size), dtype=np.float32),
            "requester_rewards": np.zeros((count, candidate_size), dtype=np.float32),
            "event_ids": np.zeros((count,), dtype=np.int64),
            "worker_ids": np.zeros((count,), dtype=np.int64),
            "positive_project_ids": np.zeros((count,), dtype=np.int64),
            "candidate_project_ids": np.zeros((count, candidate_size), dtype=np.int64),
        }
    return outputs, counts


def project_time_features(project: ProjectInfo, event_dt: datetime | None) -> tuple[float, float]:
    if event_dt is None or project.start_date is None or project.deadline is None:
        return 0.0, 0.0
    total_seconds = max((project.deadline - project.start_date).total_seconds(), 1.0)
    age_ratio = (event_dt - project.start_date).total_seconds() / total_seconds
    time_left_ratio = (project.deadline - event_dt).total_seconds() / total_seconds
    return clip01(age_ratio), clip01(time_left_ratio)


def make_features(
    *,
    worker_quality: float,
    quality_unknown: float,
    worker_hist: WorkerHistory,
    project_hist: ProjectHistory,
    worker_project_count: int,
    project: ProjectInfo,
    event_dt: datetime | None,
    active_project_count: float,
    normalizers: dict[str, float],
) -> list[float]:
    worker_submit = worker_hist.submit_count
    project_submit = project_hist.submit_count
    age_ratio, time_left_ratio = project_time_features(project, event_dt)

    category_match = 1.0 if worker_hist.category_counts.get(project.category, 0) > 0 else 0.0
    industry_match = 1.0 if worker_hist.industry_counts.get(project.industry_id, 0) > 0 else 0.0
    worker_category_preference = safe_div(
        worker_hist.category_counts.get(project.category, 0), worker_submit
    )
    worker_industry_preference = safe_div(
        worker_hist.industry_counts.get(project.industry_id, 0), worker_submit
    )

    return [
        1.0,
        clip01(worker_quality),
        quality_unknown,
        log_norm(worker_submit, normalizers["max_worker_submit_log"]),
        safe_div(worker_hist.score_sum, worker_submit) / 5.0,
        safe_div(worker_hist.winner_count, worker_submit),
        safe_div(worker_hist.finalist_count, worker_submit),
        safe_div(worker_hist.withdrawn_count, worker_submit),
        safe_div(worker_hist.award_log_sum, worker_submit) / normalizers["max_award_log"],
        project.category / normalizers["max_category"],
        project.sub_category / normalizers["max_sub_category"],
        project.industry_id / normalizers["max_industry"],
        project.award_log / normalizers["max_award_log"],
        project.tips_log / normalizers["max_tips_log"],
        log_norm(project.expected_answer_num, normalizers["max_expected_answer_log"]),
        log_norm(project.entry_count, normalizers["max_entry_count_log"]),
        log_norm(project.creative_count, normalizers["max_creative_count_log"]),
        project.average_score / 5.0,
        project.featured,
        project.assured,
        project.private_gallery,
        log_norm(project.duration_hours, normalizers["max_duration_log"]),
        age_ratio,
        time_left_ratio,
        log_norm(project_submit, normalizers["max_entry_count_log"]),
        safe_div(project_hist.score_sum, project_submit) / 5.0,
        safe_div(project_hist.winner_count, project_submit),
        safe_div(project_hist.finalist_count, project_submit),
        safe_div(project_hist.withdrawn_count, project_submit),
        category_match,
        industry_match,
        worker_category_preference,
        worker_industry_preference,
        log_norm(worker_project_count, normalizers["max_worker_submit_log"]),
        log_norm(active_project_count, normalizers["max_active_log"]),
    ]


def compute_rewards(
    *,
    candidate_project_id: int,
    positive_project_id: int,
    project: ProjectInfo,
    worker_quality: float,
    quality_unknown: float,
    category_match: float,
    industry_match: float,
    event: dict[str, str],
) -> tuple[float, float]:
    hit = 1.0 if candidate_project_id == positive_project_id else 0.0
    normalized_score = parse_float(event.get("normalized_score"))
    winner = parse_float(event.get("winner"))
    finalist = parse_float(event.get("finalist"))
    withdrawn = parse_float(event.get("withdrawn"))
    award_log = math.log1p(max(parse_float(event.get("award_value")), 0.0))
    is_original = parse_float(event.get("is_entirely_original"))

    if hit:
        worker_reward = (
            1.0 * hit
            + 0.5 * category_match
            + 0.5 * industry_match
            + 0.5 * normalized_score
            + 1.0 * finalist
            + 2.0 * winner
            + 0.5 * award_log
            - 0.2 * withdrawn
        )
        project_need_score = 1.0 - safe_div(project.entry_count, max(project.expected_answer_num, 1.0))
        requester_reward = (
            1.0 * hit
            + 1.0 * worker_quality
            + 0.8 * normalized_score
            + 1.0 * finalist
            + 2.0 * winner
            + 0.3 * is_original
            + 0.3 * clip01(project_need_score)
            - 0.5 * withdrawn
            - 0.1 * quality_unknown
        )
    else:
        worker_reward = -0.1 + 0.2 * category_match + 0.2 * industry_match
        requester_reward = 0.1 * worker_quality + 0.1 * category_match + 0.1 * industry_match

    return worker_reward, requester_reward


def update_histories(
    event: dict[str, str],
    projects: dict[int, ProjectInfo],
    worker_histories: defaultdict[int, WorkerHistory],
    project_histories: defaultdict[int, ProjectHistory],
) -> None:
    worker_id = parse_int(event.get("worker_id"), default=-1)
    project_id = parse_int(event.get("positive_project_id"), default=-1)
    if worker_id < 0 or project_id not in projects:
        return

    project = projects[project_id]
    score = parse_float(event.get("score"))
    winner = parse_int(event.get("winner"))
    finalist = parse_int(event.get("finalist"))
    withdrawn = parse_int(event.get("withdrawn"))
    award_log = math.log1p(max(parse_float(event.get("award_value")), 0.0))

    worker_hist = worker_histories[worker_id]
    worker_hist.submit_count += 1
    worker_hist.score_sum += score
    worker_hist.winner_count += winner
    worker_hist.finalist_count += finalist
    worker_hist.withdrawn_count += withdrawn
    worker_hist.award_log_sum += award_log
    worker_hist.category_counts[project.category] += 1
    worker_hist.industry_counts[project.industry_id] += 1
    worker_hist.project_counts[project_id] += 1

    project_hist = project_histories[project_id]
    project_hist.submit_count += 1
    project_hist.score_sum += score
    project_hist.winner_count += winner
    project_hist.finalist_count += finalist
    project_hist.withdrawn_count += withdrawn


def write_json(path: Path, obj: dict[str, Any] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = load_projects(processed_dir / "project.csv")
    events = load_events(processed_dir)
    if not events:
        raise ValueError("No events found. Please run src/build_events.py first.")

    candidate_size = len(events[0]["candidate_project_ids"].split("|"))
    for row in events[:1000]:
        size = len(row["candidate_project_ids"].split("|"))
        if size != candidate_size:
            raise ValueError("Inconsistent candidate size found in event files.")

    normalizers = compute_normalizers(projects, events)
    outputs, target_counts = allocate_outputs(
        events=events,
        candidate_size=candidate_size,
        feature_dim=len(FEATURE_NAMES),
        max_events_per_split=args.max_events_per_split,
    )
    write_indices = {split: 0 for split in SPLITS}
    worker_histories: defaultdict[int, WorkerHistory] = defaultdict(WorkerHistory)
    project_histories: defaultdict[int, ProjectHistory] = defaultdict(ProjectHistory)
    skipped_missing_project = 0
    skipped_by_debug_limit = 0

    for event in events:
        split = event["_split"]
        if args.max_events_per_split and write_indices[split] >= args.max_events_per_split:
            skipped_by_debug_limit += 1
            update_histories(event, projects, worker_histories, project_histories)
            continue

        worker_id = parse_int(event.get("worker_id"), default=-1)
        positive_project_id = parse_int(event.get("positive_project_id"), default=-1)
        candidate_ids = [parse_int(pid, default=-1) for pid in event["candidate_project_ids"].split("|")]
        if positive_project_id not in projects or any(pid not in projects for pid in candidate_ids):
            skipped_missing_project += 1
            update_histories(event, projects, worker_histories, project_histories)
            continue

        row_idx = write_indices[split]
        event_dt = parse_time(event.get("event_time", ""))
        worker_hist = worker_histories[worker_id]
        worker_quality = parse_float(event.get("worker_quality"))
        quality_unknown = parse_float(event.get("quality_unknown"), default=1.0)
        active_project_count = parse_float(event.get("active_project_count"))

        for action_idx, project_id in enumerate(candidate_ids):
            project = projects[project_id]
            project_hist = project_histories[project_id]
            worker_project_count = worker_hist.project_counts.get(project_id, 0)
            features = make_features(
                worker_quality=worker_quality,
                quality_unknown=quality_unknown,
                worker_hist=worker_hist,
                project_hist=project_hist,
                worker_project_count=worker_project_count,
                project=project,
                event_dt=event_dt,
                active_project_count=active_project_count,
                normalizers=normalizers,
            )
            outputs[split]["states"][row_idx, action_idx, :] = np.asarray(features, dtype=np.float32)

            category_match = features[FEATURE_NAMES.index("category_match")]
            industry_match = features[FEATURE_NAMES.index("industry_match")]
            worker_reward, requester_reward = compute_rewards(
                candidate_project_id=project_id,
                positive_project_id=positive_project_id,
                project=project,
                worker_quality=worker_quality,
                quality_unknown=quality_unknown,
                category_match=category_match,
                industry_match=industry_match,
                event=event,
            )
            outputs[split]["worker_rewards"][row_idx, action_idx] = worker_reward
            outputs[split]["requester_rewards"][row_idx, action_idx] = requester_reward

        outputs[split]["labels"][row_idx] = parse_int(event.get("label_index"))
        outputs[split]["event_ids"][row_idx] = parse_int(event.get("event_id"))
        outputs[split]["worker_ids"][row_idx] = worker_id
        outputs[split]["positive_project_ids"][row_idx] = positive_project_id
        outputs[split]["candidate_project_ids"][row_idx, :] = np.asarray(candidate_ids, dtype=np.int64)
        write_indices[split] += 1

        update_histories(event, projects, worker_histories, project_histories)

    for split in SPLITS:
        path = output_dir / f"{split}_features.npz"
        np.savez_compressed(path, **outputs[split])
        print(
            f"Wrote {write_indices[split]} {split} feature rows to {path} "
            f"with state shape {outputs[split]['states'].shape}"
        )

    stats = {
        "candidate_size": candidate_size,
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "target_counts": target_counts,
        "written_counts": write_indices,
        "skipped_missing_project": skipped_missing_project,
        "skipped_by_debug_limit": skipped_by_debug_limit,
        "normalizers": normalizers,
        "note": (
            "Dynamic histories are updated after featurizing each event, so worker/project "
            "history features only use past events."
        ),
    }
    write_json(output_dir / "feature_names.json", FEATURE_NAMES)
    write_json(output_dir / "features_stats.json", stats)
    print(f"Wrote feature metadata to {output_dir / 'features_stats.json'}")


if __name__ == "__main__":
    main()

