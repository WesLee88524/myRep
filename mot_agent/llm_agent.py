from .schemas import InvestigationReport, SuspectEvent
from .tools import ToolBox


class ReviewAgent:
    """LLM-orchestrated investigation agent (rule-based stub for V1)."""

    def __init__(self, tools: ToolBox, max_tool_calls: int = 8):
        self.tools = tools
        self.max_tool_calls = max_tool_calls

    def investigate(self, event: SuspectEvent) -> InvestigationReport:
        t1, t2 = event.time_range
        tool_trace = []
        evidence = []

        clip = self.tools.get_clip(event.track_id, max(1, t1 - 15), t2 + 15, slow_motion=0.5)
        tool_trace.append({"tool": "get_clip", "args": {"track_id": event.track_id, "t1": max(1, t1 - 15), "t2": t2 + 15, "slow_motion": 0.5}, "result": clip})

        if event.suspect_type == "FALSE_NEGATIVE_GAP":
            det = self.tools.detect_in_roi(clip["clip_ref"], [0, 0, 1920, 1080])
            tool_trace.append({"tool": "detect_in_roi", "args": {"clip_ref": clip["clip_ref"], "roi": [0, 0, 1920, 1080]}, "result": det})

            if det["num_det"] > 0:
                evidence.append({"type": "detector_support", "score": 0.78, "detail": "Detector fired in the gap window."})
                return InvestigationReport(
                    event_id=event.event_id,
                    verdict="CONFIRMED_ERROR",
                    error_type="FALSE_NEGATIVE_GAP",
                    final_confidence=0.75,
                    edit_proposals=[
                        {
                            "op": "interpolate_gap",
                            "args": {"track_id": event.track_id, "t1": t1, "t2": t2},
                        }
                    ],
                    needs_human_review=False,
                    evidence=evidence,
                    tool_trace=tool_trace,
                )

            return InvestigationReport(
                event_id=event.event_id,
                verdict="INCONCLUSIVE",
                error_type="NONE",
                final_confidence=0.5,
                needs_human_review=True,
                evidence=evidence,
                tool_trace=tool_trace,
            )

        if event.suspect_type == "ID_SWITCH":
            overlay = self.tools.overlay_tracks(clip["clip_ref"], [event.track_id])
            tool_trace.append({"tool": "overlay_tracks", "args": {"clip_ref": clip["clip_ref"], "track_ids": [event.track_id]}, "result": overlay})

            reid = self.tools.compare_reid(
                f"{event.track_id}@{max(1, t1 - 10)}-{t1}",
                f"{event.track_id}@{t2}-{t2 + 10}",
            )
            tool_trace.append({"tool": "compare_reid", "args": {"seg_a": f"{event.track_id}@{max(1, t1 - 10)}-{t1}", "seg_b": f"{event.track_id}@{t2}-{t2 + 10}"}, "result": reid})

            if reid["cosine"] < 0.5:
                evidence.append({"type": "appearance_inconsistency", "score": 0.82, "detail": "ReID similarity dropped below threshold."})
                return InvestigationReport(
                    event_id=event.event_id,
                    verdict="CONFIRMED_ERROR",
                    error_type="ID_SWITCH",
                    final_confidence=0.80,
                    edit_proposals=[
                        {
                            "op": "split_track",
                            "args": {"track_id": event.track_id, "frame": (t1 + t2) // 2},
                        }
                    ],
                    needs_human_review=False,
                    evidence=evidence,
                    tool_trace=tool_trace,
                )

            return InvestigationReport(
                event_id=event.event_id,
                verdict="NO_ERROR",
                error_type="NONE",
                final_confidence=0.70,
                needs_human_review=False,
                evidence=evidence,
                tool_trace=tool_trace,
            )

        return InvestigationReport(
            event_id=event.event_id,
            verdict="INCONCLUSIVE",
            error_type="NONE",
            final_confidence=0.5,
            needs_human_review=True,
            evidence=evidence,
            tool_trace=tool_trace,
        )
