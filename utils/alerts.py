from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime


def play_system_alert() -> None:
    try:
        import winsound

        winsound.Beep(2000, 250)
        return
    except Exception:
        pass

    # Terminal bell fallback for non-Windows environments.
    print("\a", end="")


@dataclass
class AlertManager:
    enable_sound: bool = False
    cooldown_seconds: float = 2.0
    last_alert_timestamp: float = field(default=0.0, init=False)
    total_violations: int = field(default=0, init=False)
    latest_alert_label: str | None = field(default=None, init=False)

    def trigger(self, label: str) -> str | None:
        now = time.time()
        if now - self.last_alert_timestamp < self.cooldown_seconds:
            return None

        self.last_alert_timestamp = now
        self.total_violations += 1
        self.latest_alert_label = label
        if self.enable_sound:
            play_system_alert()
        return datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
