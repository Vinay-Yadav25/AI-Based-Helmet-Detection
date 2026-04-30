# Reliability Guide

This project now includes code-level reliability upgrades for live monitoring, plus recommended training and dataset improvements.

## Implemented Runtime Enhancements

- Minimum bounding-box area filtering to suppress tiny noisy detections
- Consecutive-frame confirmation for webcam violations
- Lightweight no-helmet track matching using IoU
- Unique violation counting instead of frame-by-frame repeated counting
- Alert cooldown control
- Evidence snapshot saving for confirmed violations
- Recent event history in the Streamlit UI
- Adjustable runtime thresholds in the sidebar

Evidence is stored under `evidence/` when enabled.

## Recommended Training Upgrades

For better real-world reliability, retrain with stronger settings when hardware allows:

```bash
python train.py --data dataset/hardhat_yolo/data.yaml --model yolov8s.pt --imgsz 768 --epochs 100 --batch 2 --device 0 --workers 0 --cache false --project models --name helmet_detector_reliable
```

Recommended experiments:

- `YOLOv8s` instead of `YOLOv8n`
- `epochs=100`
- `imgsz=768`
- Evaluate on a held-out test environment, not only the validation split

## Recommended Dataset Improvements

Add more examples for:

- low light
- backlit workers
- rain or fog
- motion blur
- crowded scenes
- partially occluded helmets
- far-away workers
- different helmet colors and shapes
- false-positive backgrounds such as round objects and posters

Annotation checklist:

- helmet boxes tightly cover the visible helmet
- no-helmet boxes tightly cover the exposed head region
- no missing labels in crowded scenes
- consistent labeling for partial visibility

## Suggested Evaluation Workflow

1. Train on `train`
2. Tune on `val`
3. Keep a separate unseen `test` environment
4. Review:
   - confusion matrix
   - false positives
   - false negatives
   - missed small objects
5. Add failure cases back into the dataset and retrain
