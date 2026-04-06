from typing import Dict, List

from .schemas import Det, SuspectEvent


def propose_suspects_heuristic(
    tracks: Dict[int, List[Det]],
    video_id: str,
    gap_threshold: int = 3,
    velocity_jump_ratio: float = 3.0,
) -> List[SuspectEvent]:
    """Generate high-recall suspect events before VLM integration."""
    events: List[SuspectEvent] = []
    event_counter = 0

    for track_id, seq in tracks.items():
        if len(seq) < 2:
            continue

        # Missing detection gaps.
        for cur, nxt in zip(seq[:-1], seq[1:]):
            gap = nxt.frame - cur.frame
            if gap >= gap_threshold:
                events.append(
                    SuspectEvent(
                        event_id=f"E_{event_counter:06d}",
                        video_id=video_id,
                        track_id=track_id,
                        time_range=(cur.frame, nxt.frame),
                        suspect_type="FALSE_NEGATIVE_GAP",
                        confidence=0.65,
                        evidence_hints=[f"frame_gap={gap}"],
                    )
                )
                event_counter += 1

        # Sudden motion changes (possible ID switch / bad association).
        if len(seq) >= 3:
            for i in range(1, len(seq) - 1):
                p0, p1, p2 = seq[i - 1], seq[i], seq[i + 1]
                dt1 = max(1, p1.frame - p0.frame)
                dt2 = max(1, p2.frame - p1.frame)
                v1 = ((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5 / dt1
                v2 = ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5 / dt2

                if v1 > 0 and (v2 / v1) > velocity_jump_ratio:
                    events.append(
                        SuspectEvent(
                            event_id=f"E_{event_counter:06d}",
                            video_id=video_id,
                            track_id=track_id,
                            time_range=(max(1, p1.frame - 2), p1.frame + 2),
                            suspect_type="ID_SWITCH",
                            confidence=0.60,
                            evidence_hints=["velocity_jump"],
                        )
                    )
                    event_counter += 1

    return sorted(events, key=lambda e: e.confidence, reverse=True)
