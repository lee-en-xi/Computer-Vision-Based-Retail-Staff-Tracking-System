# 🛍️ Computer-Vision-Based Retail Staff Tracking System

**Find the staff member in an overhead camera video using only the small name tag on their chest.**

The system tells you **which frames** contain the staff member and **where** they are (x, y). It does not look at clothing colour, height or appearance, because the staff member changes clothes in the sample video and other people wear similar colours. **The name tag is the only clue.**

> Sample video: 960 x 720, 25 fps, 53.6 s, 1,341 frames, a crowded office.

---

## ✨ At a Glance

| | |
| --- | --- |
| 🎯 **Task 1** | Which frames contain the staff member? |
| 📍 **Task 2 (bonus)** | Where is the staff member, as x, y coordinates? |
| 🏷️ **Identity clue** | A small piano-key name tag, only about 17 x 11 pixels |
| 🧠 **Models** | YOLO11s tag detector (trained by me) + YOLO11m person detector + ByteTrack |
| ✅ **Tag detector score (test windows)** | Recall 0.96 · mAP50 0.95 |
| 🚫 **False-alarm check** | 0 false alarms at the 0.62 cut-off on 2 tag-free videos |

---

## 🔑 Key Features

- **Name tag only.** Works even when clothes change, so it can later run on a real camera.
- **Strict by design.** A person is called staff only when the tag scores **0.62 or higher**. A missed frame is better than naming the wrong person.
- **Keeps following when the tag is hidden.** The same person ID is followed for a few seconds, then the path is predicted from the last direction and speed, inside a limited search circle.
- **Easy to read output.** Green means the position is sure, yellow means it is estimated, and every frame shows the time, frame number and x, y.
- **Two scripts.** A full version and a simple version that draws a green box only when the tag is clearly seen.

---

## 🧩 How It Works

```
        Video frame
             │
     ┌───────┴────────┐
     ▼                ▼
 Tag detector    Person detector + ByteTrack
 (YOLO11s)       (YOLO11m, gives each person an ID)
     │                │
     └───────┬────────┘
             ▼
        Staff rules
   • tag score 0.62 or more on a person  →  STAFF (sure)
   • tag hidden, same person ID          →  keep following (sure)
             │
             ▼
   Output video with a point, circle and x, y
```

**What you see on the video**

| Mark | Meaning |
| --- | --- |
| 🟩 Thin green box + score | A name tag detected with score 0.62 or higher |
| 🟢 Green dot with a ring | Staff position is **sure** (state `tag`, `track` or `tag-only`) |
| 🟡 Yellow circle | Staff position is **estimated**, the circle grows with time since the last sure frame (state `predicted`) |
| ⬛ Black banner | Time, frame number and state |

---

## 🤖 Models Used

| Purpose | Model | Details |
| --- | --- | --- |
| 🏷️ Name-tag detector | **YOLO11s**, fine-tuned by the developer (Model B) | 9.4 M parameters, input 960 px, about 10 to 17 ms per image on a T4 GPU |
| 🧍 Person detector | **YOLO11m** (pretrained, person class only) | Input 1280 px, confidence 0.10 |
| 🔗 Person tracker | **ByteTrack** (inside Ultralytics) | track_buffer 90, match_thresh 0.9 |

**Why YOLO?** It scans the picture once, so it is fast enough for video, and it works well for tiny objects at high resolution. **Why ByteTrack?** It also uses weak, blurry detections, so a fast-moving person is not dropped as easily.

---

## ⚙️ Installation

**1. Clone the repository**

```
git clone https://github.com/lee-en-xi/Computer-Vision-Based-Retail-Staff-Tracking-System.git
cd Computer-Vision-Based-Retail-Staff-Tracking-System
```

**2. Install the packages**

```
pip install -r requirements.txt
```

> 💡 A GPU is recommended. On a laptop without a GPU the person model is slow. In that case set `PERSON_IMGSZ = 960` at the top of the script.

---

## ▶️ How to Run

**1. Open the script** (`run_video.py` for the full version, `run_video_simple.py` for the simple one).

**2. Change one line** at the top to your video path:

```python
VIDEO = "path/to/your_video.mp4"
```

The tag model is found automatically at `weights/name_tag_best_model.pt`, so keep that folder.

**3. Run it**

```
python run_video.py
```

**4. Get the results.** The script writes **one video** next to your video and prints a summary.

| Script | What it does | Output video |
| --- | --- | --- |
| `run_video.py` | Full version: tag + person tracking + hidden-tag handling | `<video name>_annotated.mp4` |
| `run_video_simple.py` | Simple version: staff box only when the tag scores 0.62 or higher, no prediction | `<video name>_green.mp4` |

The first run downloads `yolo11m.pt` by itself, so it needs internet once. The scripts also run in Google Colab: paste one into a cell, set `VIDEO` to a path in your Drive, and run.

**Printed summary (full version):** total frames, frames with a clearly seen tag, the count of each state (`tag`, `tag-only`, `track`, `predicted`, `none`), and a table of the moments with start and end times.

---

## 📊 Results

### Sample video

| Item | Result |
| --- | --- |
| Frames where the tag is clearly seen (0.62 or higher) | **209 of 1,341 (15.6%)**, in 16 moments between 13.6 s and 48.4 s |
| Frames with a staff position, including estimated ones (notebook pipeline with gap filling) | **762 of 1,341** |

In the frames between the 16 moments, the staff member is still in the room but the tag is not readable, so the position is estimated.

### Tag detector: Model A vs Model B

Test windows (never used for training or choosing the best epoch, 28 tags):

| Metric | Model A (real only) | Model B (real + synthetic) |
| --- | --- | --- |
| Recall: real tags found | 0.86 | **0.96** |
| mAP50: overall detection score | 0.87 | **0.95** |
| mAP50-95: how exactly the box fits | 0.44 | **0.62** |
| Precision: boxes that were real tags | **0.91** | 0.87 |

Validation windows (27 tags):

| Metric | Model A | Model B |
| --- | --- | --- |
| Precision | 0.99 | 1.00 |
| Recall | 0.85 | 0.93 |
| mAP50 | 0.94 | 0.98 |
| mAP50-95 | 0.48 | 0.67 |

**Why Model B?** It finds more of the real tags (recall 0.96 vs 0.86) and has a higher overall score (mAP50 0.95 vs 0.87). Its precision is slightly lower, which is handled by accepting only scores of **0.62 or higher**.

### Why 0.62?

Every wrong tag detection I saw in the sample video scored 0.61 or lower, so the cut-off sits just above that. It was set on one video and should be rechecked for each new camera.

### False-alarm test on tag-free videos

Two office videos from the internet with people but no staff tag. Nothing should be flagged.

| Video | Frames | Frames at 0.62 or higher | Highest score |
| --- | --- | --- | --- |
| Tag-free test 1 | 306 | **0** | 0.00 |
| Tag-free test 2 | 391 | **0** | 0.61 |

In test 2, the strongest wrong detections (0.58 to 0.61) are at about 8.7 to 9.7 s and sit on a person, probably a similar-looking badge or clothing pattern. The margin to the cut-off is only 0.01, so more tag-free footage would make it safer.

---

## 🏋️ How the Tag Detector Was Trained

1. **Real data.** I cut the clearest views of the tag out of the video, removed the background, and labelled the tag in Roboflow.
2. **Synthetic data.** The cut-out tag is pasted onto people's torsos in video frames at a realistic size (8 to 32 px wide), with matching blur. Moving people are favoured, and a paste is accepted only if at least 85% of the tag lies on the person. Frames come only from the training part of the video (before 37.8 s) and skip every moment where the real tag is visible. The final run produced 687 synthetic images.
3. **Split by time, not at random.** Neighbouring frames look almost identical, so a random split would leak answers. Whole time windows were held out:
   - Validation: 38.9 to 39.4 s, 40.6 to 41.2 s, 42.4 to 43.0 s
   - Test: 44.3 to 44.9 s, 46.2 to 46.8 s, 48.0 to 48.6 s
4. **Training set.** Real: 128 unique frames (67 with a tag, 61 without), each used 3 times = 384 images. Synthetic: 250 kept at random out of 687. **Model B total: 634 images.**
5. **Two models compared** (same YOLO11s settings, input 960 px, batch 16):

   | | Model A | Model B (chosen) |
   | --- | --- | --- |
   | Training data | Real only: 384 | Real + synthetic: 634 |
   | Epochs | Stopped early at 43 of 80 (best epoch 18) | All 80 |

The full process is in `notebook/01_train_name_tag_detector.ipynb`.

---

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── run_video.py        
├── run_video_simple.py  
├── notebook/
│   ├── 01_train_name_tag_detector.ipynb  
│   └── 02_test_on_video.ipynb             
├── weights/
│   └── name_tag_best_model.pt             
└── sample_output/        
```


