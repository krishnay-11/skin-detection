# Deep Learning Based Multi-Condition Facial Skin Analysis and Recommendation System

A college project that analyses a face photo, detects several visible skin conditions, estimates their severity, explains what the model focused on, gives general skincare tips and tracks changes over time.

> **Disclaimer:** this tool gives general skincare suggestions only. It is not a medical diagnosis.

## Features
1. Detects multiple conditions in one image: acne, eyebag, wrinkle, darkcircle, pigmentation, spots
2. Shows the approximate location of each condition (bounding boxes)
3. Estimates severity (mild, moderate, severe) from the count and the area covered
4. Gives personalized general skincare recommendations (rule-based)
5. Explains model focus with an EigenCAM heatmap (a gradient-free, Grad-CAM-style method)
6. Tracks changes between scans using a local SQLite history

## Tech
Python, YOLOv8s (Ultralytics, PyTorch), OpenCV, Streamlit, SQLite. Trained on Google Colab (T4 GPU).

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

The trained weights are in `models/best.pt`.

## Datasets (all downloaded from Roboflow Universe, CC BY 4.0)
- ACNE+PE (spark002/acne-pe)
- face (wise-man/face-a0vwp)
- picgment (hdh-yqcg5/picgment)
- Acne-rural (flickr/acne-rural)

Classes were remapped to six common labels and merged. The acne-heavy images were undersampled to reduce class imbalance. The datasets are not included in this repo; please cite the original authors if you reuse them.

## Results (held-out test split, 603 images)
| Model | mAP50 | mAP50-95 | Recall |
|---|---|---|---|
| Baseline (ACNE+PE only) | 0.238 | 0.095 | 0.278 |
| v3_merged (4 datasets) | 0.324 | 0.135 | 0.412 |

Per-class mAP50-95 for v3_merged: Acne 0.130, Eyebag 0.093, Wrinkle 0.113, darkcircle 0.335, pigmentation 0.026, spots 0.112.

## Limitations
- Pigmentation and spots are weak because of small and inconsistent labels
- Severity and recommendations are rule-based; the thresholds are not validated on labelled severity data
- Datasets are community-labelled and may be biased in skin tone, lighting and image quality
- Progress tracking needs similar lighting and pose between scans
- EigenCAM shows overall feature focus, not the cause of one specific prediction

## Authors
Add the real names and roles of everyone who worked on this project.
