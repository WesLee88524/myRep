import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List

from .editor import TrackEditor
from .llm_agent import ReviewAgent
from .parser import index_tracks, load_mot_txt
from .suspects import propose_suspects_heuristic
from .tools import ToolBox


def run_pipeline(mot_txt: str, video_id: str = "UNKNOWN") -> Dict[str, Any]:
    dets = load_mot_txt(mot_txt)
    tracks = index_tracks(dets)

    suspects = propose_suspects_heuristic(tracks, video_id=video_id)
    tools = ToolBox()
    agent = ReviewAgent(tools=tools)
    editor = TrackEditor(tracks=tracks)

    reports: List[Dict[str, Any]] = []
    applied_edits = 0

    for event in suspects:
        report = agent.investigate(event)
        reports.append(asdict(report))

        if report.verdict == "CONFIRMED_ERROR":
            for proposal in report.edit_proposals:
                if editor.apply(proposal):
                    applied_edits += 1

    return {
        "num_dets": len(dets),
        "num_tracks": len(tracks),
        "num_suspects": len(suspects),
        "num_reports": len(reports),
        "num_applied_edits": applied_edits,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOT error-review agent pipeline.")
    parser.add_argument("mot_txt", type=str, help="Path to MOTChallenge txt file")
    parser.add_argument("--video-id", type=str, default="UNKNOWN", help="Video identifier")
    parser.add_argument("--out", type=str, default="agent_report.json", help="Output JSON report path")
    args = parser.parse_args()

    result = run_pipeline(args.mot_txt, video_id=args.video_id)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
