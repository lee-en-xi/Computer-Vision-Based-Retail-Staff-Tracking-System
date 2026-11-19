# Computer-Vision-Based Retail Staff Tracking System

Find the staff member in an overhead-camera video using only the small piano-key **name tag** on their chest, and report **which frames** contain the staff member (Task 1) and **where** they are, as x, y coordinates (Task 2, bonus).

The system does not use clothing colour, height or appearance, because the staff member changes clothes in the sample video and other people wear similar colours. The name tag is the only identity clue.

---

## 1. Results at a glance

| Item | Result |
| --- | --- |
| Tag detector (Model B), test windows | Recall 0.96, mAP50 0.95, mAP50-95 0.62 |
| Sample video: frames where the tag is clearly seen (score 0.62 or higher) | 209 of 1,341 frames (15.6%), in 16 moments |
| Sample video: frames with a staff position, including estimated ones (offline notebook pipeline) | 762 of 1,341 frames |
| Tag-free negative test 1 (306 frames) | 0 frames flagged, highest score 0.00 |
| Tag-free negative test 2 (391 frames) | 0 frames flagged, highest score 0.61 (just under the 0.62 cut-off) |

End-to-end precision and recall against a full hand-labelled answer key are **not measured yet** (see Limitations).

---

## 2. Repository contents

```
README.md                 this file
requirements.txt          Python packages
.gitignore
run_video.py              full pipeline, frame by frame (tag + person tracking + hidden-tag handling)
run_video_simple.py       simple version: no tracking, no prediction
notebooks/
  01_train_name_tag_detector.ipynb   data, synthetic images, training models A and B, evaluation
  02_test_on_video.ipynb             tracking and testing on the video, negative tests
weights/
  name_tag_best_model.pt  trained name-tag detector (Model B)
sample_output/            example results (CSV, demo video)
```

The notebooks are for **reading** how the work was done. The two `run_video*.py` scripts are what you **run** on a new video.

---

## 3. Models used

| Purpose | Model | Notes |
| --- | --- | --- |
| Name-tag detector | **YOLO11s** (Ultralytics), fine-tuned by me | 9.4 M parameters, 21.4 GFLOPs, input size 960 px. About 10 to 17 ms per image on a Tesla T4 |
| Person detector | **YOLO11m** (pretrained COCO, person class only) | Input size 1280 px, confidence 0.10 |
| Person tracker | **ByteTrack** (built into Ultralytics) | Custom settings: track_buffer 90, match_thresh 0.9, track_high_thresh 0.25, track_low_thresh 0.10 |

**Why YOLO:** it scans the whole picture in one pass, so it is fast enough for video, works well for small objects at high resolution (the tag is only about 17 x 11 px in a 960 x 720 frame), and trains with little data. **Why YOLO11:** the newer generation, reported by its authors to be more accurate than YOLOv8 at a similar size.

**Why ByteTrack:** it also uses weak, blurry detections, so a fast-moving person is not dropped as easily.

---

## 4. How the system works

1. **Tag detector** looks for the name tag in every frame and gives each detection a confidence score.
2. **Person tracker** finds every person and gives each an ID number that follows them across frames.
3. **Staff rules:**
   - A tag with score **0.62 or higher** on a person confirms that person as staff (state `tag`).
   - A strong tag with no person box around it (person missed, for example when blurred) still counts, shown on the tag itself (state `tag-only`).
   - When the tag is hidden, the same person ID keeps the staff label for up to 3 seconds (state `track`).
   - If that person is lost, the position is predicted from the last direction and speed for up to 1.5 seconds, inside a search circle that grows with time and is capped at 160 px (state `predicted`).
4. **Output:** a point, with a circle of uncertainty when the position is estimated, plus the frame time and x, y coordinates.

On screen: a thin green box and score on every strong tag, a green dot with a ring when the position is sure, a yellow circle when it is estimated, and a black banner with time, frame number and state.

---

## 5. How the tag detector was trained

1. **Real data.** I cut the clearest views of the tag out of the sample video, removed the background, and labelled the tag in Roboflow by drawing a box around it. There were too few real examples.
2. **Synthetic data.** A person-segmentation step finds people in video frames, and a cut-out tag is pasted onto each person's torso at a realistic size (8 to 32 px wide), with blur matched to the tag type. Moving people are favoured, and a paste is accepted only if at least 85% of the tag lies on the person. Each pasted tag comes with its box already known. My first version put tags on tables and hands, so I restricted them to the body. Synthetic frames come only from the **training part of the video** (before 37.8 s), skip every moment where the real tag is visible, use every 3rd frame and make up to 3 variants per frame. The final run produced 687 synthetic images.
3. **Split by time, not at random.** Neighbouring frames look almost identical, so a random split would leak answers between sets. Whole time windows were held out:
   - Validation: 38.9 to 39.4 s, 40.6 to 41.2 s, 42.4 to 43.0 s (27 images, 27 tags)
   - Test: 44.3 to 44.9 s, 46.2 to 46.8 s, 48.0 to 48.6 s (28 images, 28 tags)
   - Training: the other tag windows
4. **Training set composition.**
   - **Real images:** 128 unique labelled frames (67 with a tag, 61 background frames with no tag). Each was used 3 times (the original plus 2 copies, with augmentation applied during training), giving **384 images**.
   - **Synthetic images:** I kept **250** of the 687 generated, chosen at random with a fixed seed. Keeping only 250 stops the synthetic images from drowning out the real ones, so real data is still about 60% of the training set.
   - **Model B total: 634 training images.** Before I trimmed the synthetic set, the training folder held 1,071 images (384 real + 687 synthetic); the models were trained on the trimmed set of 634.

5. **Two models compared** (both start from pretrained YOLO11s and use identical settings):

   | | Model A | Model B (chosen) |
   | --- | --- | --- |
   | Training data | Real images only: 384 | Real + synthetic: 634 (384 real + 250 synthetic) |
   | Epochs | Stopped early at epoch 43 of 80 (best epoch: 18), 0.19 hours | All 80 epochs, 0.53 hours |
   | Validation set | 27 images | 27 images |

   Shared settings: Ultralytics 8.4.215 on a Tesla T4 GPU, input size 960 px, batch size 16, automatic optimiser (AdamW), early-stopping patience 25, single class `name_tag`, augmentation with mosaic, flips, rotation up to 180 degrees, scale and brightness changes.

---

## 6. Tag detector metrics

**Test windows** (never used for training or choosing the best epoch, 28 tags):

| Metric | Model A (real only) | Model B (real + synthetic) |
| --- | --- | --- |
| Recall: real tags found | 0.86 | **0.96** |
| mAP50: overall detection score | 0.87 | **0.95** |
| mAP50-95: how exactly the box fits | 0.44 | **0.62** |
| Precision: boxes that were real tags | **0.91** | 0.87 |

**Validation windows** (27 tags):

| Metric | Model A | Model B |
| --- | --- | --- |
| Precision | 0.99 | 1.00 |
| Recall | 0.85 | 0.93 |
| mAP50 | 0.94 | 0.98 |
| mAP50-95 | 0.48 | 0.67 |

**Why Model B:** it finds more of the real tags (recall 0.96 vs 0.86) and has a higher overall score (mAP50 0.95 vs 0.87). Its precision is slightly lower (0.87 vs 0.91). That is handled by accepting only detections with a score of **0.62 or higher**.

The test sets are small (one tag is about 3.5 percentage points) and come from the same video, so they show how the model handles unseen moments of this video, not other cameras.

---

## 7. Why the threshold is 0.62

In the sample video, every wrong tag detection I saw scored 0.61 or lower, so the cut-off is set just above that. It is a strict cut-off: it prefers missing a frame over naming the wrong person. It was set on one video and should be rechecked for each new camera.

---

## 8. Testing on tag-free videos (false-alarm check)

Two office videos from the internet with people but no staff tag. Nothing should be flagged.

| Video | Frames | Frames at 0.62 or higher | Highest score |
| --- | --- | --- | --- |
| Tag-free test 1 | 306 | 0 | 0.00 |
| Tag-free test 2 | 391 | 0 | 0.61 |

In test 2, 32 frames scored 0.45 or higher. The strongest ones (0.58 to 0.61) are at about 8.7 to 9.7 s and sit on a person, probably a similar-looking badge or clothing pattern. The margin to the 0.62 cut-off is only 0.01, so more tag-free footage would make the cut-off safer.

---

## 9. Sample video results

The sample video is 960 x 720, 25 fps, 53.6 s, 1,341 frames.

- **Task 1:** the tag is clearly seen (score 0.62 or higher) in **209 frames (15.6%)**, in 16 moments between 13.6 s and 48.4 s. In the frames between these moments the staff member is still in the room, but the tag is not readable, so the position is estimated.
- **Task 2:** the offline notebook pipeline gives a position for **762 of 1,341 frames**, which includes estimated frames. It uses extra gap filling that needs future frames.
- The live-style `run_video.py` does not use future frames, so expect fewer estimated frames than the notebook.

---

## 10. How to run

**Install**

```
pip install -r requirements.txt
```

**Run the full version** (tag + person tracking + hidden-tag handling)

1. Open `run_video.py`.
2. Change the `VIDEO` line at the top to your video path. The script finds the tag model at `weights/name_tag_best_model.pt` next to it, so keep that folder structure.
3. Run:

```
python run_video.py
```

It writes **one video**, `<video name>_annotated.mp4`, next to your video, and prints a summary: frame counts, frames with a clearly seen tag, the count per state, and a table of the moments with start and end times.

**Run the simple version** (green boxes only when the tag scores 0.62 or higher, no prediction)

```
python run_video_simple.py
```

It writes `<video name>_green.mp4` and prints a short summary.

The person model `yolo11m.pt` downloads itself on the first run, so the first run needs internet. A GPU is recommended: on a CPU-only laptop the larger person model is slow, so consider lowering `PERSON_IMGSZ` to 960.

The same scripts also run in Google Colab: paste a script into a cell, set `VIDEO` to a path in your Drive, and run it.

**Retrain the tag detector (optional).** `notebooks/01_train_name_tag_detector.ipynb` shows the full training process. It needs a Roboflow account and your own API key, and the original dataset is not stored in this repository.

---

## 11. Limitations

- **Hidden tag, no exact position.** Without a visible tag the output is an estimated area, not a precise point.
- **False alarms are still possible.** One tag-free video scored 0.61, just under the cut-off.
- **Fast, blurry walking.** If the person finder misses the staff while she turns a corner and the tag is hidden, the estimate can lag behind.
- **Small training base.** All real examples come from one video, one tag design and one camera angle, so new lighting or a new office may need retraining.
- **One staff member only.** With two staff, the system would need a separate tracker for each.
- **Staff are not told apart.** The system knows "this is staff", not who. Two staff crossing paths could swap labels.
- **Recorded video.** The notebook pipeline uses a short look-ahead; a live camera needs a delay of 1 to 2 seconds or the simpler version, and enough computing power for speed.
- **Settings tuned on one video.** The cut-off and search radius must be rechecked for each new camera.
- **Small test and no answer key.** The detector was tested on only 28 tags, and the estimated positions were not checked against hand-labelled frames.

**Next steps:** label the staff frames by hand to measure real precision and recall, test on more tag-free videos, add hard negatives to training, and add a motion-based check for fast walkers that the person finder misses.