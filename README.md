# Thermal-Based Human Detection & Behavior Analysis

A deep learning pipeline that recognizes human activities (walking, jumping jacks, push-ups, tai chi, and more) from **thermal-like video**, and separately detects and tracks people in real thermal footage — without requiring thermal camera hardware for training.

## Problem

Thermal cameras are valuable for low-light and privacy-preserving human activity recognition, but large labeled thermal-video datasets are scarce and thermal hardware is expensive. This project asks: *can a model trained on simulated thermal data, derived from ordinary RGB video, still classify human behavior accurately — and generalize toward real thermal footage?*

## Pipeline

The project has two parallel tracks: **behavior classification** (CNN-LSTM, trained on simulated thermal video) and **person detection & tracking** (YOLO, trained on real FLIR thermal data).

### Behavior Classification

| Script | Purpose |
|---|---|
| `03_make_thermal_frames.py` | Converts RGB video (UCF101) into thermal-like frames: grayscale → CLAHE contrast enhancement → Gaussian blur → normalization, with a quality filter to skip blank/corrupt frames |
| `04_build_sequences.py` | Groups frames into 16-frame sequences per video for spatiotemporal learning |
| `05_train_cnn_lstm.py` | Trains the CNN-LSTM classifier (TimeDistributed CNN blocks + LSTM + dense head) using a memory-mapped data loader to handle large sequence arrays without loading everything into RAM |
| `06_evaluate_test.py` | Evaluates the trained model on the held-out test set; outputs a classification report and confusion matrix |
| `07_live_prediction.py` | Runs real-time behavior prediction from a webcam feed, with majority-vote smoothing to reduce flicker |
| `08_video_behavior_demo.py` | Runs behavior prediction on a pre-recorded video file, outputting a side-by-side RGB/thermal annotated demo video |

### Person Detection & Tracking (FLIR / YOLO)

| Script | Purpose |
|---|---|
| `10_flir_coco_to_yolo.py` | Converts the FLIR ADAS dataset's COCO-format annotations into YOLO label format, keeping only the "person" class |
| `09_flir_sequence_tracking.py` | Runs YOLO person detection + tracking over a folder of thermal frames, classifying each tracked person as Moving / Stationary / Loitering based on centroid displacement over time |
| `11_custom_video_person_motion.py` | Same detection/tracking/behavior logic, applied to a custom RGB or thermal video file |
| `12_flir_v2_testing.py` | An iterated version of the tracking logic with hysteresis-based behavior switching (harder to enter "Moving," easier to stay in it) and motion normalized by bounding-box size, for more stable behavior labels on pedestrians |

### Utility

- `check_frames_structure.py` — sanity-checks that the extracted frame folders have the expected split/class/video structure before building sequences

## Model Architecture

CNN-LSTM: 3 TimeDistributed convolutional blocks (32 → 64 → 128 filters, each with max pooling + batch norm) → global average pooling → LSTM(128) → dense classification head.

## Results

| Metric | Value |
|---|---|
| Classification accuracy (12 activity classes) | **86.07%** |
| Training sequences | 22,992 |
| Test sequences | 3,797 |
| YOLO detection precision (FLIR) | 0.694 |
| YOLO detection mAP@50 | 0.652 |

## Key Findings

- Visually distinct activities classified reliably; visually similar movements were the main source of confusion in the confusion matrix.
- **Prediction flickering** — brief, incorrect label switches between adjacent frames — was a real challenge for live/video inference, addressed with majority-vote smoothing (behavior classification) and hysteresis-based state switching (person tracking).
- The gap between simulated thermal data (training) and real thermal footage (FLIR) is the main limitation for production deployment.

## Tech Stack

`Python` `TensorFlow/Keras` `OpenCV` `NumPy` `Ultralytics YOLO` `scikit-learn`

## Datasets

- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) — RGB action recognition dataset (subset of classes used for behavior classification)
- [FLIR ADAS Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/) — real thermal imagery, used for person detection/tracking

Datasets, trained model weights, and generated outputs are not included in this repo (see `.gitignore`) — they're large and, in FLIR's case, subject to FLIR's own license terms. Scripts expect them under `data/` and `models/` locally; see each script's `argparse` options or config section for exact expected paths.

## Running It

```bash
pip install -r requirements.txt

# Behavior classification pipeline
python 03_make_thermal_frames.py
python 04_build_sequences.py
python 05_train_cnn_lstm.py
python 06_evaluate_test.py
python 08_video_behavior_demo.py --video path/to/video.mp4

# FLIR detection/tracking pipeline
python 10_flir_coco_to_yolo.py --flir-root path/to/FLIR_ADAS
# (train a YOLO model on the converted dataset separately, via Ultralytics)
python 09_flir_sequence_tracking.py --model path/to/best.pt --frames path/to/frames
```

## Future Work

- Fine-tune on real thermal camera footage to close the simulation-to-real domain gap
- Explore Bi-LSTM, 3D CNN, or transformer-based temporal models for behavior classification
- Add confidence-based temporal smoothing to further reduce prediction flickering
- Deploy on edge hardware (Raspberry Pi / Jetson) for real-time on-device inference

---
*Master's project, University of Massachusetts Dartmouth.*
