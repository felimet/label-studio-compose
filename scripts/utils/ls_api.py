from __future__ import annotations

import sys
from typing import Iterator

import requests


class LabelStudioAPI:
    """Thin wrapper around Label Studio REST API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            }
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 10)
        resp = self._session.get(f"{self.base_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp

    def _post(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 10)
        resp = self._session.post(f"{self.base_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp

    def _delete(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 10)
        resp = self._session.delete(f"{self.base_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp

    # ── Public API ────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Label Studio is reachable and API key is valid."""
        try:
            self._get("/api/projects/", params={"page_size": 1})
            return True
        except Exception:
            return False

    def get_project(self, project_id: int) -> dict:
        """Fetch project metadata including label_config."""
        return self._get(f"/api/projects/{project_id}/").json()

    def list_tasks(
        self, project_id: int, page_size: int = 100
    ) -> Iterator[dict]:
        """Yield all tasks for a project, following pagination (next link)."""
        url: str | None = (
            f"{self.base_url}/api/projects/{project_id}/tasks/"
            f"?page_size={page_size}"
        )
        while url:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # API may return list directly or wrapped in {"results": [...], "next": ...}
            if isinstance(data, list):
                yield from data
                break
            else:
                yield from data.get("results", [])
                next_link = data.get("next")
                url = next_link if next_link else None

    def get_task(self, task_id: int) -> dict:
        """Fetch a single task with fresh annotation count."""
        return self._get(f"/api/tasks/{task_id}/").json()

    def list_predictions(self, task_id: int) -> list[dict]:
        """List all predictions for a task."""
        return self._get(
            "/api/predictions/", params={"task": task_id}
        ).json()

    def delete_prediction(self, prediction_id: int) -> None:
        """Delete a single prediction by ID."""
        try:
            self._delete(f"/api/predictions/{prediction_id}/")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                pass  # already deleted
            else:
                raise

    def delete_cli_predictions(self, task_id: int, model_version: str) -> int:
        """Delete all predictions for a task that match model_version exactly.

        Only predictions with an exact model_version match are deleted.
        Predictions from other sources are NOT touched.
        Returns the number of predictions deleted.
        """
        try:
            predictions = self.list_predictions(task_id)
        except Exception:
            return 0

        # Handle both list response and paginated response
        if isinstance(predictions, dict):
            predictions = predictions.get("results", [])

        deleted = 0
        for p in predictions:
            if p.get("model_version") == model_version:
                self.delete_prediction(p["id"])
                deleted += 1
        return deleted

    def create_prediction(
        self,
        task_id: int,
        result: list,
        score: float,
        model_version: str,
    ) -> dict:
        """Write a prediction for a task.

        score: prediction confidence (0.0–1.0).
        model_version: MUST be passed explicitly by the caller (CLI_MODEL_VERSION_SAM3
        or CLI_MODEL_VERSION_SAM21). Never read model_version from the ML backend
        response — SAM2.1 does not set it.
        """
        payload = {
            "task": task_id,
            "result": result,
            "score": score,
            "model_version": model_version,
        }
        return self._post("/api/predictions/", json=payload).json()
