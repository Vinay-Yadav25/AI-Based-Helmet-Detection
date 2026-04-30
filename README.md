# Real-Time Safety Helmet Detection

End-to-end safety helmet detection project using YOLOv8, OpenCV, and Streamlit.

## Features

- Train a custom YOLO-format dataset with classes `helmet` and `no_helmet`
- Validate trained weights with precision, recall, mAP@50, and mAP@50-95
- Run inference on image, video, or webcam
- Streamlit UI for uploads and live webcam mode
- Color-coded bounding boxes
- Alerting for `no_helmet` detections
- Violation counting and FPS display

## Project Structure

```text
helmet_detection/
├── dataset/
├── models/
├── utils/
├── app.py
├── detect.py
├── train.py
└── requirements.txt
```

## Setup

```bash
cd helmet_detection
pip install -r requirements.txt
```

## Dataset

Place your custom YOLO-format dataset under `dataset/` and keep the class ids:

- `0` -> `helmet`
- `1` -> `no_helmet`

The default dataset config is `dataset/data.yaml`.

If you are using the included Hardhat dataset in Pascal VOC XML format, convert it first:

```bash
python convert_dataset.py --source dataset/Hardhat --output dataset/hardhat_yolo
```

This project maps:

- `helmet` -> `helmet`
- `head` -> `no_helmet`

and ignores other labels such as `person` and `others`.

## Training

```bash
python train.py --data dataset/data.yaml --model yolov8s.pt --imgsz 640 --epochs 50 --batch 4
```

`YOLOv8s` is now the default training backbone because it is more reliable than `YOLOv8n` while still being practical on modest GPUs. If you want to push accuracy further and your GPU can handle it, try `yolov8m.pt`.

Training artifacts are stored under `models/helmet_detector/`.

## CLI Detection

Webcam:

```bash
python detect.py --model models/helmet_detector/weights/best.pt --source 0
```

Image:

```bash
python detect.py --model models/helmet_detector/weights/best.pt --source path/to/image.jpg --save
```

Video:

```bash
python detect.py --model models/helmet_detector/weights/best.pt --source path/to/video.mp4 --save
```

Press `q` to stop webcam or video inference.

## Streamlit App

```bash
streamlit run app.py
```

Sidebar options:

- Upload image/video
- Webcam mode
- Confidence threshold
- Optional alert sound
- Model path override
