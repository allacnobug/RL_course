#!/usr/bin/env python3
"""Preprocess raw crowdsourcing data for the RL recommendation project.

The script reads the original CSV/JSON-text files under data/ and writes
structured CSV files for later event construction, feature engineering, and
DQN training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENTRY_FILE_RE = re.compile(r"entry_(\d+)_(\d+)\.txt$")
PROJECT_FILE_RE = re.compile(r"project_(\d+)\.txt$")


PROJECT_FIELDS = [
    "project_id",
    "project_list_answer_num",
    "category",
    "sub_category",
    "industry",
    "industry_id",
    "status",
    "start_date",
    "deadline",
    "created_at",
    "updated_at",
    "duration_hours",
    "entry_count",
    "creative_count",
    "average_score",
    "client_feedback",
    "watchers_count",
    "total_awards",
    "award_log",
    "total_package_tips",
    "tips_log",
    "featured",
    "assured",
    "private_gallery",
    "search_exclusion",
    "nda_required",
    "package_name",
    "entry_type",
    "is_one_to_one",
    "is_legacy",
]

ENTRY_FIELDS = [
    "project_id",
    "entry_id",
    "entry_number",
    "worker_id",
    "author_username",
    "entry_created_at",
    "updated_at",
    "winner",
    "finalist",
    "withdrawn",
    "withdrawn_by_admin",
    "eliminated",
    "award_value",
    "offer_value",
    "tip_value",
    "revision_count",
    "score",
    "normalized_score",
    "max_revision_number",
    "is_entirely_original",
    "entry_type",
    "feedback_count",
]

WORKER_FIELDS = [
    "worker_id",
    "worker_quality",
    "quality_unknown",
    "raw_worker_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw crowdsourcing data into project/entry/worker CSV files."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/processed"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately when a project or entry file cannot be parsed.",
    )
    return parser.parse_args()


def parse_time(value: Any) -> str:
    """Parse common ISO timestamps and return UTC ISO string.

    Empty or invalid values return an empty string. The downstream code can then
    decide whether to drop or impute the row.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return ""
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


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


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> int:
    return int(bool(value))


def hours_between(start_iso: str, end_iso: str) -> float:
    if not start_iso or not end_iso:
        return 0.0
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except ValueError:
        return 0.0
    return max((end - start).total_seconds() / 3600.0, 0.0)


def read_json(path: Path, strict: bool, errors: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - error is recorded for report/debugging.
        if strict:
            raise
        errors.append({"file": str(path), "error": repr(exc)})
        return None


def read_project_list(path: Path) -> dict[int, int]:
    project_answer_num: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            project_id = parse_int(row[0], default=-1)
            answer_num = parse_int(row[1], default=0)
            if project_id >= 0:
                project_answer_num[project_id] = answer_num
    return project_answer_num


def read_worker_quality(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            worker_id = parse_int(row[0], default=-1)
            raw_quality = parse_float(row[1], default=-1.0)
            if worker_id < 0:
                continue
            quality_unknown = raw_quality < 0
            worker_quality = 0.0 if quality_unknown else raw_quality / 100.0
            rows.append(
                {
                    "worker_id": worker_id,
                    "worker_quality": worker_quality,
                    "quality_unknown": int(quality_unknown),
                    "raw_worker_quality": raw_quality,
                }
            )
        if header is None:
            return rows
    return rows


def clean_projects(
    data_dir: Path,
    project_answer_num: dict[int, int],
    strict: bool,
    errors: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    project_rows: list[dict[str, Any]] = []
    industry_to_id: dict[str, int] = {}
    project_dir = data_dir / "project"

    for path in sorted(project_dir.glob("project_*.txt")):
        match = PROJECT_FILE_RE.match(path.name)
        if not match:
            continue
        project_id = int(match.group(1))
        obj = read_json(path, strict=strict, errors=errors)
        if obj is None:
            continue

        industry = str(obj.get("industry") or "unknown")
        if industry not in industry_to_id:
            industry_to_id[industry] = len(industry_to_id)

        start_date = parse_time(obj.get("start_date"))
        deadline = parse_time(obj.get("deadline"))
        total_awards = parse_float(obj.get("total_awards"))
        total_package_tips = parse_float(obj.get("total_package_tips"))

        project_rows.append(
            {
                "project_id": project_id,
                "project_list_answer_num": project_answer_num.get(project_id, 0),
                "category": parse_int(obj.get("category")),
                "sub_category": parse_int(obj.get("sub_category")),
                "industry": industry,
                "industry_id": industry_to_id[industry],
                "status": obj.get("status") or "",
                "start_date": start_date,
                "deadline": deadline,
                "created_at": parse_time(obj.get("created_at")),
                "updated_at": parse_time(obj.get("updated_at")),
                "duration_hours": hours_between(start_date, deadline),
                "entry_count": parse_int(obj.get("entry_count")),
                "creative_count": parse_int(obj.get("creative_count")),
                "average_score": parse_float(obj.get("average_score")),
                "client_feedback": parse_float(obj.get("client_feedback")),
                "watchers_count": parse_int(obj.get("watchers_count")),
                "total_awards": total_awards,
                "award_log": math.log1p(max(total_awards, 0.0)),
                "total_package_tips": total_package_tips,
                "tips_log": math.log1p(max(total_package_tips, 0.0)),
                "featured": parse_bool(obj.get("featured")),
                "assured": parse_bool(obj.get("assured")),
                "private_gallery": parse_bool(obj.get("private_gallery")),
                "search_exclusion": parse_bool(obj.get("search_exclusion")),
                "nda_required": parse_bool(obj.get("nda_required")),
                "package_name": obj.get("package_name") or "",
                "entry_type": obj.get("entry_type") or "",
                "is_one_to_one": parse_bool(obj.get("is_one_to_one")),
                "is_legacy": parse_bool(obj.get("is_legacy")),
            }
        )

    return project_rows, industry_to_id


def extract_revision_features(entry: dict[str, Any]) -> dict[str, Any]:
    revisions = entry.get("revisions") or []
    scores: list[float] = []
    revision_numbers: list[int] = []
    originality_values: list[int] = []

    for revision in revisions:
        scores.append(parse_float(revision.get("score")))
        revision_numbers.append(parse_int(revision.get("revision_number")))
        if revision.get("is_entirely_original") is not None:
            originality_values.append(parse_bool(revision.get("is_entirely_original")))

    score = max(scores) if scores else 0.0
    max_revision_number = max(revision_numbers) if revision_numbers else 0
    is_entirely_original = max(originality_values) if originality_values else 0

    return {
        "revision_count": len(revisions),
        "score": score,
        "normalized_score": score / 5.0 if score > 0 else 0.0,
        "max_revision_number": max_revision_number,
        "is_entirely_original": is_entirely_original,
    }


def clean_entries(
    data_dir: Path,
    strict: bool,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    entry_rows: list[dict[str, Any]] = []
    entry_dir = data_dir / "entry"

    for path in sorted(entry_dir.glob("entry_*.txt")):
        match = ENTRY_FILE_RE.match(path.name)
        if not match:
            continue
        project_id = int(match.group(1))
        obj = read_json(path, strict=strict, errors=errors)
        if obj is None:
            continue

        for entry in obj.get("results") or []:
            revision_features = extract_revision_features(entry)
            feedback = entry.get("entry_feedback") or {}
            worker_id = parse_int(entry.get("worker", entry.get("author")), default=-1)

            entry_rows.append(
                {
                    "project_id": project_id,
                    "entry_id": parse_int(entry.get("id")),
                    "entry_number": parse_int(entry.get("entry_number")),
                    "worker_id": worker_id,
                    "author_username": entry.get("author_username") or "",
                    "entry_created_at": parse_time(entry.get("entry_created_at")),
                    "updated_at": parse_time(entry.get("updated_at")),
                    "winner": parse_bool(entry.get("winner")),
                    "finalist": parse_bool(entry.get("finalist")),
                    "withdrawn": parse_bool(entry.get("withdrawn")),
                    "withdrawn_by_admin": parse_bool(entry.get("withdrawn_by_admin")),
                    "eliminated": parse_bool(entry.get("eliminated")),
                    "award_value": parse_float(entry.get("award_value")),
                    "offer_value": parse_float(entry.get("offer_value")),
                    "tip_value": parse_float(entry.get("tip_value")),
                    "entry_type": entry.get("entry_type") or "",
                    "feedback_count": len(feedback),
                    **revision_features,
                }
            )

    entry_rows.sort(key=lambda row: (row["entry_created_at"], row["project_id"], row["entry_number"]))
    return entry_rows


def add_missing_entry_workers(
    worker_rows: list[dict[str, Any]], entry_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    existing = {int(row["worker_id"]) for row in worker_rows}
    for worker_id in sorted({int(row["worker_id"]) for row in entry_rows if int(row["worker_id"]) >= 0}):
        if worker_id not in existing:
            worker_rows.append(
                {
                    "worker_id": worker_id,
                    "worker_quality": 0.0,
                    "quality_unknown": 1,
                    "raw_worker_quality": -1.0,
                }
            )
    worker_rows.sort(key=lambda row: int(row["worker_id"]))
    return worker_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_stats(
    project_rows: list[dict[str, Any]],
    entry_rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
    industry_to_id: dict[str, int],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    entry_times = [row["entry_created_at"] for row in entry_rows if row["entry_created_at"]]
    project_status = Counter(row["status"] for row in project_rows)
    project_industries = Counter(row["industry"] for row in project_rows)
    entry_project_ids = {int(row["project_id"]) for row in entry_rows}
    project_ids = {int(row["project_id"]) for row in project_rows}

    worker_entry_count: defaultdict[int, int] = defaultdict(int)
    for row in entry_rows:
        worker_id = int(row["worker_id"])
        if worker_id >= 0:
            worker_entry_count[worker_id] += 1

    return {
        "project_count": len(project_rows),
        "entry_count": len(entry_rows),
        "worker_count": len(worker_rows),
        "entry_worker_count": len(worker_entry_count),
        "known_quality_worker_count": sum(1 for row in worker_rows if int(row["quality_unknown"]) == 0),
        "unknown_quality_worker_count": sum(1 for row in worker_rows if int(row["quality_unknown"]) == 1),
        "winner_entry_count": sum(int(row["winner"]) for row in entry_rows),
        "finalist_entry_count": sum(int(row["finalist"]) for row in entry_rows),
        "withdrawn_entry_count": sum(int(row["withdrawn"]) for row in entry_rows),
        "missing_entry_time_count": sum(1 for row in entry_rows if not row["entry_created_at"]),
        "missing_project_time_count": sum(
            1 for row in project_rows if not row["start_date"] or not row["deadline"]
        ),
        "entry_time_min": min(entry_times) if entry_times else "",
        "entry_time_max": max(entry_times) if entry_times else "",
        "project_without_entries_count": len(project_ids - entry_project_ids),
        "entry_project_missing_project_file_count": len(entry_project_ids - project_ids),
        "industry_count": len(industry_to_id),
        "top_project_status": project_status.most_common(20),
        "top_industries": project_industries.most_common(20),
        "parse_error_count": len(errors),
        "parse_errors_sample": errors[:20],
    }


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    errors: list[dict[str, str]] = []

    project_list_path = data_dir / "project_list.csv"
    worker_quality_path = data_dir / "worker_quality.csv"
    if not project_list_path.exists():
        raise FileNotFoundError(f"Missing project list: {project_list_path}")
    if not worker_quality_path.exists():
        raise FileNotFoundError(f"Missing worker quality file: {worker_quality_path}")

    project_answer_num = read_project_list(project_list_path)
    worker_rows = read_worker_quality(worker_quality_path)
    project_rows, industry_to_id = clean_projects(
        data_dir=data_dir,
        project_answer_num=project_answer_num,
        strict=args.strict,
        errors=errors,
    )
    entry_rows = clean_entries(data_dir=data_dir, strict=args.strict, errors=errors)
    worker_rows = add_missing_entry_workers(worker_rows, entry_rows)

    write_csv(output_dir / "project.csv", project_rows, PROJECT_FIELDS)
    write_csv(output_dir / "entry.csv", entry_rows, ENTRY_FIELDS)
    write_csv(output_dir / "worker.csv", worker_rows, WORKER_FIELDS)
    write_stats(output_dir / "industry_map.json", industry_to_id)
    write_stats(
        output_dir / "stats.json",
        make_stats(project_rows, entry_rows, worker_rows, industry_to_id, errors),
    )

    print(f"Wrote {len(project_rows)} projects to {output_dir / 'project.csv'}")
    print(f"Wrote {len(entry_rows)} entries to {output_dir / 'entry.csv'}")
    print(f"Wrote {len(worker_rows)} workers to {output_dir / 'worker.csv'}")
    print(f"Wrote stats to {output_dir / 'stats.json'}")
    if errors:
        print(f"Warning: {len(errors)} files failed to parse. See stats.json for samples.")


if __name__ == "__main__":
    main()

