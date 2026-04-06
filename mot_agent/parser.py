import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .schemas import Det


def load_mot_txt(path: str) -> List[Det]:
    """Load MOTChallenge-style txt rows into Det objects."""
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"MOT txt not found: {path}")

    dets: List[Det] = []
    with txt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            tid = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            conf = float(row[6]) if len(row) > 6 else 1.0
            cls = int(float(row[7])) if len(row) > 7 else -1
            vis = float(row[8]) if len(row) > 8 else -1.0
            dets.append(
                Det(
                    frame=frame,
                    track_id=tid,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    conf=conf,
                    cls=cls,
                    vis=vis,
                )
            )
    return dets


def index_tracks(dets: List[Det]) -> Dict[int, List[Det]]:
    tracks = defaultdict(list)
    for det in dets:
        tracks[det.track_id].append(det)

    indexed = dict(tracks)
    for track_id in indexed:
        indexed[track_id].sort(key=lambda d: d.frame)
    return indexed
