# MOT Error-Review Agent (Python, MOTChallenge txt)

A minimal pipeline that follows the three-step design:

1. **Suspect proposal**: high-recall heuristic events (`FALSE_NEGATIVE_GAP`, `ID_SWITCH`) to bootstrap VLM integration.
2. **Investigation agent**: tool-orchestrated review stage (V1 uses deterministic stubs that mirror LLM+tools behavior).
3. **Track correction**: atomic edit operations with rollback support.

## Quickstart

```bash
python -m mot_agent.main /path/to/seq.txt --video-id MOT17-04 --out report.json
```

## MOT txt format

Expected row format (MOTChallenge style):

```text
frame,id,x,y,w,h,conf,cls,vis
```

At minimum, columns 1-7 are required.

## Project layout

- `mot_agent/schemas.py`: core dataclasses.
- `mot_agent/parser.py`: MOT txt parser + track indexing.
- `mot_agent/suspects.py`: heuristic suspect event generator.
- `mot_agent/tools.py`: tool API stubs for clip/reid/detect checks.
- `mot_agent/llm_agent.py`: investigation policy (LLM placeholder).
- `mot_agent/editor.py`: track edit operations + rollback.
- `mot_agent/main.py`: end-to-end CLI pipeline.

## Notes

- This is a V1 scaffold intended for plugging in real backends:
  - VLM candidate generation
  - ReID model inference
  - detector-in-ROI
  - OpenCV-based visualization/zoom/enhancement
