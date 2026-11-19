"""
Simple version: no tracking, no prediction.
In frames where the name tag scores >= TAG_STRONG on a person, everything is drawn in GREEN:
  - a thin box on the name tag, with its score
  - a box on the staff member, labelled STAFF (x,y) with the tag's coordinates
  - a black banner at the top-left: time, frame number and staff state
Change VIDEO below, run, and everything else stays the same.
Output: one video (<video name>_green.mp4, next to the video) and a short printed summary. No other files.
"""
import os, cv2, numpy as np, pandas as pd
from ultralytics import YOLO

# ============ CHANGE THIS ============
VIDEO = "/content/drive/MyDrive/staff_detection/copy_sample_video.mp4"
# =====================================

try:
    HERE = os.path.dirname(os.path.abspath(__file__))     # folder of this script
except NameError:                                          # running inside a notebook cell
    HERE = os.getcwd()
TAG_WEIGHTS = os.path.join(HERE, "weights", "name_tag_best_model.pt")   # Model B (name-tag detector)
PERSON_WEIGHTS = "yolo11m.pt"       # downloads itself the first time
TAG_STRONG = 0.62                   # tag score needed to confirm staff
GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    out_video = os.path.splitext(os.path.abspath(VIDEO))[0] + "_green.mp4"
    tag_model, person_model = YOLO(TAG_WEIGHTS), YOLO(PERSON_WEIGHTS)

    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"cannot open video: {VIDEO}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        found = False
        tr = tag_model(frame, conf=TAG_STRONG, imgsz=960, verbose=False)[0]          # only strong tags are returned
        if len(tr.boxes):
            pr = person_model(frame, classes=[0], conf=0.25, imgsz=1280, verbose=False)[0]   # people (run only when needed)
            people = pr.boxes.xyxy.tolist()
            for c, (x1, y1, x2, y2) in zip(tr.boxes.conf.tolist(), tr.boxes.xyxy.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                holders = [b for b in people if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
                if holders:                                                           # tag must be on a person
                    b = min(holders, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, 2)       # staff box
                    cv2.putText(frame, f"STAFF ({int(cx)},{int(cy)})", (int(b[0]), max(14, int(b[1]) - 10)), FONT, 0.6, GREEN, 2)
                    cv2.rectangle(frame, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), GREEN, 1)  # name-tag box
                    cv2.putText(frame, f"tag {c:.2f}", (int(x1) - 8, int(y1) - 12), FONT, 0.4, GREEN, 1)
                    found = True
                    rows.append(dict(frame=n, time_s=round((n - 1) / fps, 2), tag_conf=round(c, 2),
                                     x=int(cx), y=int(cy)))
        cv2.rectangle(frame, (0, 0), (360, 26), (0, 0, 0), -1)                              # banner
        cv2.putText(frame, f"t={(n - 1) / fps:5.2f}s  frame {n - 1}  staff: {'tag' if found else 'none'}",
                    (5, 18), FONT, 0.5, (255, 255, 255), 1)
        writer.write(frame)
        if n % 100 == 0:
            print(f"frame {n} ...")
    cap.release(); writer.release()

    d = pd.DataFrame(rows)
    nf = d.frame.nunique() if len(d) else 0
    print(f"\nvideo: {n} frames ({n / fps:.1f} s) | frames with a green box: {nf} ({100 * nf / max(n, 1):.1f}%)")
    print("video saved to:", out_video)


if __name__ == "__main__":
    main()