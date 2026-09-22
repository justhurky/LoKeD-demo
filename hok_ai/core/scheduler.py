from __future__ import annotations

from dataclasses import dataclass

from .db import Database


@dataclass(slots=True)
class DeadlineScheduler:
    db: Database
    enabled: bool = True
    scheduler: object | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            self.db.log("warning", "APScheduler nincs telepitve, hatarido figyeles kihagyva.")
            return

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(self.check_deadlines, "interval", minutes=30, id="deadline_check", replace_existing=True)
        scheduler.start()
        self.scheduler = scheduler
        self.db.log("info", "Hatarido figyelo elindult.")

    def stop(self) -> None:
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None

    def check_deadlines(self) -> None:
        dashboard = self.db.dashboard()
        urgent = dashboard.get("urgent_tasks", [])
        if urgent:
            self.db.log("info", "Surgos vagy hianyos feladatok vannak.", {"count": len(urgent)})
