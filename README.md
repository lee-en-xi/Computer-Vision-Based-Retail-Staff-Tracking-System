# Computer-Vision-Based-Retail-Staff-Tracking-System (Overhead Camera)

Detects a small, blurry staff name tag in overhead ceiling-camera footage of a
crowded office, then tracks the wearer so they stay identified when the tag is hidden.

> **Status: work in progress.** Tag detection and tracking work on the development clip.
> Evaluation on other cameras and videos is still pending.

## Problem
A person counts as staff if they wear a name tag. From a ceiling camera the tag is
tiny (median ~17 × 11 px at 960 × 720), often blurry, and often hidden by the
person's head and shoulders, so tag visibility is not the same as staff presence.

## Approach
1. **Tag detector:** YOLO11s fine-tuned on real labelled frames plus synthetic images.
2. **Synthetic data:** tag cut-outs pasted onto person-segmentation masks (YOLO11-seg),
   with random rotation, scale, perspective, brightness and blur.
3. **Tracking:** YOLO11 person detection + ByteTrack. A strong tag detection
   (conf ≥ 0.62) marks that person as staff, and the staff keeps their label while
   the tag is hidden.
4. **Output:** annotated video, per-frame CSV (frame, staff present, XY coordinates).

## Results (held-out time windows of one clip)
| Model | Split | Precision | Recall | mAP50 |
|---|---|---|---|---|
| A: real only | test | 0.91 | 0.86 | 0.87 |
| B: real + synthetic | test | 0.87 | 0.96 | 0.95 |

**Caveats:** the test split has only 28 images, all from one camera. Tag crops were
taken from the whole video, so these scores are optimistic. Generalization to other
cameras is untested.

## Data handling
Train/valid/test were split by time window, not randomly, to avoid near-duplicate
frames leaking across splits. The video and dataset are not included (privacy).

## Run it
```bash
pip install -r requirements.txt
python src/track_staff.py --video your_video.mp4 --weights weights/best.pt --out results/
```

## Limitations
- Evaluated on a single clip
- The tag is often hidden from overhead, so staff identity relies on tracking
- Re-identification after the person leaves the view is not solved
- Staff and tag crops are from the same scene, so synthetic data may be too similar

## Roadmap
- [x] Real-only vs real+synthetic ablation
- [x] Tag-triggered person tracking
- [ ] Offline back-fill of earlier frames
- [ ] Timeline plot (staff present vs tag visible)
- [ ] Single-command CLI
- [ ] Test on additional videos

## Tech
Python, PyTorch, Ultralytics YOLO11, OpenCV, NumPy, Roboflow, Google Colab
