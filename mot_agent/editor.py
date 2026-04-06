from copy import deepcopy
from typing import Any, Dict, List

from .schemas import Det


class TrackEditor:
    """Apply atomic edits with undo support (logic kept simple for V1)."""

    def __init__(self, tracks: Dict[int, List[Det]]):
        self.tracks = tracks
        self.history: List[Dict[str, Any]] = []

    def apply(self, proposal: Dict[str, Any]) -> bool:
        op = proposal.get("op")
        args = proposal.get("args", {})

        if op == "interpolate_gap":
            return self._interpolate_gap(args)
        if op == "split_track":
            return self._split_track(args)
        return False

    def _interpolate_gap(self, args: Dict[str, Any]) -> bool:
        track_id = args["track_id"]
        t1 = args["t1"]
        t2 = args["t2"]
        seq = self.tracks.get(track_id)
        if not seq:
            return False

        before = deepcopy(seq)
        by_frame = {d.frame: d for d in seq}
        if t1 not in by_frame or t2 not in by_frame or t2 <= t1:
            return False

        d1 = by_frame[t1]
        d2 = by_frame[t2]
        missing_frames = [f for f in range(t1 + 1, t2) if f not in by_frame]

        for f in missing_frames:
            alpha = (f - t1) / (t2 - t1)
            seq.append(
                Det(
                    frame=f,
                    track_id=track_id,
                    x=d1.x * (1 - alpha) + d2.x * alpha,
                    y=d1.y * (1 - alpha) + d2.y * alpha,
                    w=d1.w * (1 - alpha) + d2.w * alpha,
                    h=d1.h * (1 - alpha) + d2.h * alpha,
                    conf=min(d1.conf, d2.conf) * 0.9,
                    cls=d1.cls,
                    vis=min(d1.vis, d2.vis),
                )
            )

        seq.sort(key=lambda d: d.frame)
        self.history.append({"op": "interpolate_gap", "track_id": track_id, "before": before})
        return True

    def _split_track(self, args: Dict[str, Any]) -> bool:
        track_id = args["track_id"]
        split_frame = args["frame"]
        seq = self.tracks.get(track_id)
        if not seq:
            return False

        left = [d for d in seq if d.frame <= split_frame]
        right = [d for d in seq if d.frame > split_frame]
        if not left or not right:
            return False

        before_tracks = deepcopy(self.tracks)
        new_id = max(self.tracks.keys(), default=0) + 1
        for d in right:
            d.track_id = new_id

        self.tracks[track_id] = left
        self.tracks[new_id] = right
        self.history.append({"op": "split_track", "before": before_tracks})
        return True

    def rollback_last(self) -> bool:
        if not self.history:
            return False
        record = self.history.pop()

        if record["op"] == "interpolate_gap":
            self.tracks[record["track_id"]] = record["before"]
            return True

        if record["op"] == "split_track":
            self.tracks.clear()
            self.tracks.update(record["before"])
            return True

        return False
