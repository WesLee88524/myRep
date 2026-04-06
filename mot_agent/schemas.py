from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Det:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int = -1
    vis: float = -1.0

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class SuspectEvent:
    event_id: str
    video_id: str
    track_id: int
    time_range: Tuple[int, int]
    suspect_type: str
    confidence: float
    evidence_hints: List[str] = field(default_factory=list)


@dataclass
class InvestigationReport:
    event_id: str
    verdict: str
    error_type: str
    final_confidence: float
    edit_proposals: List[Dict[str, Any]] = field(default_factory=list)
    needs_human_review: bool = False
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
