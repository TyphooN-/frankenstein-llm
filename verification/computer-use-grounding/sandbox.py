#!/usr/bin/env python3
"""A read-only simulation of a fixture screen, for state readback without side effects.

The gate must prove that a chosen action would have done the right thing. Doing
that on the real desktop is forbidden -- and would be reckless -- so the model's
action is applied to a pure-Python model of the same pixels the model saw.
Because the control rectangles come from the fixture ground truth, "the click
landed on Open Evidence" is decided by the same geometry that drew the button.

Nothing here synthesizes input, touches a display server, or writes outside the
evidence directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SandboxScreen:
    truth: dict
    clicks: list = field(default_factory=list)
    events: list = field(default_factory=list)
    last_clicked: str | None = None
    focused_field: str | None = None
    field_values: dict = field(default_factory=dict)
    checkbox_checked: bool | None = None
    rejected: list = field(default_factory=list)

    def __post_init__(self) -> None:
        for entry in self.truth.get("fields", []):
            self.field_values[entry["label"]] = entry["value"]
        checkbox = self.truth.get("checkbox")
        if checkbox:
            self.checkbox_checked = checkbox["checked"]

    # -- geometry ---------------------------------------------------------
    def _targets(self) -> list[dict]:
        targets = [{"kind": "control", **c} for c in self.truth.get("controls", [])]
        targets += [{"kind": "field", **f} for f in self.truth.get("fields", [])]
        checkbox = self.truth.get("checkbox")
        if checkbox:
            targets.append({"kind": "checkbox", **checkbox})
        return targets

    def hit(self, point: tuple[int, int] | None) -> dict | None:
        """Topmost control containing the point, or None for a miss."""
        if point is None:
            return None
        x, y = point
        for target in self._targets():
            x0, y0, x1, y1 = target["box"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return target
        return None

    def in_bounds(self, point: tuple[int, int] | None) -> bool:
        if point is None:
            return False
        width, height = self.truth["size"]
        return 0 <= point[0] < width and 0 <= point[1] < height

    # -- transitions ------------------------------------------------------
    def apply(self, verb: str | None, point: tuple[int, int] | None = None,
              content: str | None = None) -> dict:
        """Apply one action and return the event it produced. Never raises on bad input."""
        event = {"verb": verb, "point": list(point) if point else None,
                 "content": content, "accepted": False, "reason": None, "target": None}

        if verb is None:
            event["reason"] = "no parseable action"
            self.events.append(event)
            return event

        if verb in {"wait", "finished"}:
            event["accepted"] = True
            event["reason"] = f"non-grounded action {verb}"
            self.events.append(event)
            return event

        if verb == "type":
            if self.focused_field is None:
                event["reason"] = "type with no focused field"
            else:
                self.field_values[self.focused_field] = content or ""
                event.update(accepted=True, target=self.focused_field)
            self.events.append(event)
            return event

        if verb == "hotkey":
            event["reason"] = "hotkey has no sandbox effect"
            self.events.append(event)
            return event

        if not self.in_bounds(point):
            event["reason"] = "coordinate outside screen bounds"
            self.events.append(event)
            return event

        target = self.hit(point)
        if target is None:
            event["reason"] = "click landed on no control"
            self.events.append(event)
            return event

        event["target"] = target["label"]
        if target["kind"] == "control" and not target.get("enabled", True):
            # A disabled control absorbs the click: the agent believes it acted
            # and nothing happened. Treating this as success would hide a real
            # failure mode, so it is recorded as a rejection.
            event["reason"] = "control is disabled"
            self.rejected.append(target["label"])
            self.events.append(event)
            return event

        event["accepted"] = True
        if target["kind"] == "control":
            self.last_clicked = target["label"]
            self.clicks.append(target["label"])
        elif target["kind"] == "field":
            self.focused_field = target["label"]
        elif target["kind"] == "checkbox":
            self.checkbox_checked = not self.checkbox_checked
        self.events.append(event)
        return event

    def state(self) -> dict:
        return {
            "last_clicked": self.last_clicked,
            "clicks": list(self.clicks),
            "focused_field": self.focused_field,
            "field_values": dict(self.field_values),
            "checkbox_checked": self.checkbox_checked,
            "rejected_disabled": list(self.rejected),
            "event_count": len(self.events),
        }
