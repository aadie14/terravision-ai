"""Bundled demonstration image pairs (``samples/manifest.json``) with recorded provenance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.app.config import Settings
from backend.app.errors import AppError, NotFoundError

log = logging.getLogger("terravision.samples")
_ROLES = ("t1", "t2", "ground_truth")


class SampleService:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.samples_path
        self._samples: dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        self._samples = {}
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            log.warning("no sample manifest found at %s", manifest)
            return
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8-sig"))["samples"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            log.error("sample manifest unreadable: %s", exc)
            return
        for entry in entries:
            if all((self.root / entry[r]).is_file() for r in ("t1", "t2")):
                self._samples[entry["id"]] = entry
            else:
                log.warning("sample %s skipped: files missing", entry.get("id"))

    def list(self) -> list[dict]:
        return [self.public(s) for s in self._samples.values()]

    @staticmethod
    def public(sample: dict) -> dict:
        return {
            "id": sample["id"], "name": sample["name"], "description": sample.get("description", ""),
            "dataset": sample.get("dataset"), "split": sample.get("split"),
            "has_ground_truth": bool(sample.get("ground_truth")),
            "georeferenced": bool(sample.get("georeferenced", False)),
            "resolution_note": sample.get("resolution_note"),
            "domain": sample.get("domain", "vhr"),
            "license": sample.get("license"), "attribution": sample.get("attribution"),
            "source_url": sample.get("source_url"),
            "t1_date": sample.get("t1_date"), "t2_date": sample.get("t2_date"),
        }

    def get(self, sample_id: str) -> dict:
        if sample_id not in self._samples:
            raise NotFoundError("Sample", sample_id)
        return self._samples[sample_id]

    def path(self, sample_id: str, role: str) -> Path:
        sample = self.get(sample_id)
        if role not in _ROLES:
            raise AppError("UNKNOWN_SAMPLE_FILE", f"'{role}' is not one of {list(_ROLES)}", status=404)
        relative = sample.get(role)
        if not relative:
            raise AppError("NO_GROUND_TRUTH", "This sample has no ground-truth mask", status=404)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise NotFoundError("Sample file", f"{sample_id}/{role}")
        return path
