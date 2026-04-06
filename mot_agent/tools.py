from typing import Any, Dict, List


class ToolBox:
    """Stub tool APIs; replace internals with OpenCV/VLM/ReID backends."""

    def get_clip(
        self,
        track_id: int,
        t1: int,
        t2: int,
        fps: int = 30,
        slow_motion: float = 1.0,
    ) -> Dict[str, Any]:
        return {
            "ok": True,
            "clip_ref": f"clip_tid{track_id}_{t1}_{t2}",
            "fps": fps,
            "slow_motion": slow_motion,
        }

    def zoom_roi(self, clip_ref: str, roi: List[int], scale: float = 2.0) -> Dict[str, Any]:
        return {"ok": True, "zoom_ref": f"{clip_ref}_zoom_{scale}", "roi": roi}

    def enhance_clip(self, clip_ref: str, mode: str = "sharpen") -> Dict[str, Any]:
        return {"ok": True, "enhanced_ref": f"{clip_ref}_{mode}"}

    def overlay_tracks(self, clip_ref: str, track_ids: List[int]) -> Dict[str, Any]:
        return {"ok": True, "overlay_ref": f"{clip_ref}_overlay", "track_ids": track_ids}

    def compare_reid(self, seg_a: str, seg_b: str) -> Dict[str, Any]:
        # Placeholder: production should call real ReID embeddings.
        return {"ok": True, "cosine": 0.42, "seg_a": seg_a, "seg_b": seg_b}

    def motion_consistency_check(self, track_id: int, t1: int, t2: int) -> Dict[str, Any]:
        return {"ok": True, "score": 0.75, "track_id": track_id, "t1": t1, "t2": t2}

    def detect_in_roi(self, clip_ref: str, roi: List[int]) -> Dict[str, Any]:
        # Placeholder detector output.
        return {"ok": True, "num_det": 5, "roi": roi}
