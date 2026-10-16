"""Persistent, bounded local training jobs with isolated candidate artifacts."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from uuid import uuid4

from .config import CLASS_NAMES, PROJECT_ROOT
from .data_collection import atomic_json


class ModelLab:
    def __init__(self, dataset_root: Path, root: Path | None = None):
        self.dataset_root = Path(dataset_root).resolve()
        self.root = Path(root or PROJECT_ROOT / "artifacts/model_lab").resolve()
        self._lock = threading.RLock()
        self._jobs = {}
        self._processes = {}
        self._closed = False
        if self.root.is_dir():
            for path in self.root.glob("*/job.json"):
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                    if not re.fullmatch(r"[a-f0-9]{32}", job["id"]) or path.parent.name != job["id"]:
                        continue
                    if job["status"] in {"running", "queued"}:
                        job.update(status="failed", error="Backend restarted before this job completed.")
                    self._jobs[job["id"]] = job
                except (OSError, ValueError, KeyError):
                    continue

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid model job ID")
        directory = (self.root / job_id).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Invalid model job path")
        return directory

    def _save(self, job):
        atomic_json(self._path(job["id"]) / "job.json", job)

    def status(self):
        dependencies = {name: importlib.util.find_spec(name) is not None
                        for name in ("sklearn", "onnx", "onnxruntime", "pandas")}
        with self._lock:
            ordered = sorted(self._jobs, key=lambda key: self._jobs[key].get("created_at", ""), reverse=True)
            return {"available": all(dependencies.values()), "dataset_root": str(self.dataset_root),
                    "class_names": [*CLASS_NAMES, "no_gesture"],
                    "capabilities": {"dependencies": dependencies, "candidate_only": True,
                                     "context_model_training": False},
                    "jobs": [self.get(job_id) for job_id in ordered[:50]]}

    def get(self, job_id):
        with self._lock:
            self._path(job_id)
            if job_id not in self._jobs:
                raise ValueError("Model job was not found")
            job = dict(self._jobs[job_id])
            job["logs"] = list(job.get("logs", []))
            job["candidate_available"] = job["kind"] == "train" and job["status"] == "succeeded" and (self._path(job_id) / "candidate.onnx").is_file()
            return job

    def candidate_path(self, job_id):
        if not self.get(job_id)["candidate_available"]:
            raise ValueError("Select a completed training candidate")
        directory = self._path(job_id)
        contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
        import hashlib
        path = directory / "candidate.onnx"
        if contract.get("class_names") != CLASS_NAMES or contract.get("feature_count") != 76 or hashlib.sha256(path.read_bytes()).hexdigest() != contract.get("sha256"):
            raise ValueError("Candidate checksum or twelve-command contract is invalid")
        return path

    def start(self, kind, epochs=50, candidate_job_id=None):
        if kind not in {"audit", "train", "evaluate"}:
            raise ValueError("Unknown model job kind")
        if not 1 <= int(epochs) <= 500:
            raise ValueError("epochs must be between 1 and 500")
        with self._lock:
            if self._closed:
                raise ValueError("The model lab is shutting down")
            if any(job["status"] in {"running", "queued"} for job in self._jobs.values()):
                raise ValueError("A model job is already running. Wait or cancel it first.")
            if kind == "evaluate":
                self.candidate_path(candidate_job_id)
            job_id = uuid4().hex
            job = {"id": job_id, "kind": kind, "status": "queued", "epochs": int(epochs),
                   "candidate_job_id": candidate_job_id, "created_at": datetime.now(timezone.utc).isoformat(),
                   "logs": [], "result": None, "error": None}
            self._save(job)
            self._jobs[job_id] = job
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return self.get(job_id)

    def _run(self, job_id):
        process = None
        try:
            with self._lock:
                job = self._jobs[job_id]
                if self._closed or job["status"] == "cancelled":
                    return
                command = [sys.executable, "-u", "-m", "scripts.run_model_lab_job", "--kind", job["kind"],
                           "--dataset", str(self.dataset_root), "--output", str(self._path(job_id)),
                           "--epochs", str(job["epochs"])]
                if job["candidate_job_id"]:
                    command += ["--candidate", str(self._path(job["candidate_job_id"]))]
                environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                   MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", **options)
                self._processes[job_id] = process
                job["status"] = "running"
                self._save(job)
            for line in process.stdout:
                with self._lock:
                    job["logs"] = (job["logs"] + [line.rstrip()])[-400:]
                    self._save(job)
            code = process.wait()
            with self._lock:
                if job["status"] != "cancelled":
                    if code:
                        detail = next((line for line in reversed(job["logs"]) if line.strip()), "")
                        job.update(status="failed", error=detail or f"Worker exited with code {code}. See the job log.")
                    else:
                        result = json.loads((self._path(job_id) / "result.json").read_text(encoding="utf-8"))
                        job.update(status="succeeded", result=result)
                self._save(job)
        except Exception as error:
            with self._lock:
                job = self._jobs[job_id]
                if job["status"] != "cancelled":
                    job.update(status="failed", error=str(error))
                    self._save(job)
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
            if process and process.poll() is None:
                process.terminate()

    def cancel(self, job_id):
        with self._lock:
            self.get(job_id)
            job = self._jobs[job_id]
            if job["status"] in {"running", "queued"}:
                job["status"] = "cancelled"
                process = self._processes.get(job_id)
                if process and process.poll() is None:
                    process.terminate()
                self._save(job)
            return self.get(job_id)

    def close(self):
        with self._lock:
            self._closed = True
            for job_id, job in self._jobs.items():
                if job["status"] in {"running", "queued"}:
                    self.cancel(job_id)
