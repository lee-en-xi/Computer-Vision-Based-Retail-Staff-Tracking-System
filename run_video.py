"""
Staff finder: name-tag detector (model B) + person detector with ByteTrack.
Change VIDEO below, run, and everything else stays the same.

Output: one annotated video (<video name>_annotated.mp4, saved next to the video) and a short printed summary.
No CSV or other files are written.
"""
import os, tempfile, cv2, numpy as np, pandas as pd
from collections import deque
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
TAG_STRONG = 0.62                   # a tag at or above this score confirms staff
TAG_IMGSZ, PERSON_IMGSZ = 960, 1280
HOLD_S = 3.0                        # keep labelling the same person this long after the tag was last seen
PRED_S = 1.5                        # if that person is lost, predict her path for at most this long
R0, V_UNC, R_MAX = 25, 15, 160      # estimated circle: minimum radius, growth per frame since the last sure frame, cap (pixels)
DECAY = 0.90                        # prediction slows down each frame
GAP_FRAMES = 5                      # tag sightings closer than this are merged into one segment

TRACKER_YAML = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.9
fuse_score: True
"""


def inside(px, py, b):
    return b[0] <= px <= b[2] and b[1] <= py <= b[3]


def main():
    out_video = os.path.splitext(os.path.abspath(VIDEO))[0] + "_annotated.mp4"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)      # tracker settings (temporary file)
    tmp.write(TRACKER_YAML); tmp.close(); yaml_path = tmp.name

    tag_model, person_model = YOLO(TAG_WEIGHTS), YOLO(PERSON_WEIGHTS)
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"cannot open video: {VIDEO}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    staff_id, rel, last_tag_n, last_seen_n = None, (0.5, 0.3), -10**9, -10**9
    hist, last_pos, vel = deque(maxlen=6), None, np.zeros(2)
    rows, n = [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        # 1) people with ID numbers (ByteTrack)
        pr = person_model.track(frame, persist=True, tracker=yaml_path, classes=[0], conf=0.10,
                                imgsz=PERSON_IMGSZ, verbose=False)[0]
        people = {}
        if pr.boxes is not None and pr.boxes.id is not None:
            for tid, xy in zip(pr.boxes.id.int().tolist(), pr.boxes.xyxy.tolist()):
                people[tid] = xy

        # 2) the name tag (only the strongest one counts)
        tr = tag_model(frame, conf=0.30, imgsz=TAG_IMGSZ, verbose=False)[0]
        strong_tags = [(float(c), b) for c, b in zip(tr.boxes.conf.tolist(), tr.boxes.xyxy.tolist()) if c >= TAG_STRONG]
        tag = None
        if len(tr.boxes):
            i = int(np.argmax(tr.boxes.conf.cpu().numpy()))
            c = float(tr.boxes.conf[i]); x1, y1, x2, y2 = tr.boxes.xyxy[i].tolist()
            tag = (c, (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2))

        # 3) decide the staff state for this frame
        state, cx, cy, radius, conf = "none", None, None, 0, 0.0
        holder = None
        if tag and tag[0] >= TAG_STRONG:                       # strong tag must sit on a person
            cands = [(t, b) for t, b in people.items() if inside(tag[1], tag[2], b)]
            if cands:
                holder = min(cands, key=lambda tb: (tb[1][2] - tb[1][0]) * (tb[1][3] - tb[1][1]))
        if holder:
            staff_id, (tid, b) = holder[0], holder
            rel = ((tag[1] - b[0]) / max(1, b[2] - b[0]), (tag[2] - b[1]) / max(1, b[3] - b[1]))
            state, cx, cy, radius, conf = "tag", tag[1], tag[2], 0, tag[0]
            last_tag_n = n
        elif tag and tag[0] >= TAG_STRONG:                      # strong tag but no person box around it (person missed)
            state, cx, cy, radius, conf = "tag-only", tag[1], tag[2], 0, tag[0]
        elif staff_id is not None and staff_id in people and (n - last_tag_n) <= HOLD_S * fps:
            b = people[staff_id]                                # same person, tag hidden
            state, cx, cy = "track", b[0] + rel[0] * (b[2] - b[0]), b[1] + rel[1] * (b[3] - b[1])
            radius = 0
        elif staff_id is not None and staff_id not in people and last_pos is not None \
                and (n - last_seen_n) <= PRED_S * fps and (n - last_tag_n) <= (HOLD_S + PRED_S) * fps:
            k = n - last_seen_n                                 # person lost: predict from last speed
            step = vel * sum(DECAY ** j for j in range(1, k + 1))
            px, py = last_pos[0] + step[0], last_pos[1] + step[1]
            if 0 <= px < W and 0 <= py < H:
                state, cx, cy, radius = "predicted", px, py, min(R_MAX, R0 + V_UNC * k)

        if state in ("tag", "track", "tag-only"):
            last_seen_n = n; hist.append((cx, cy))
            last_pos = (cx, cy)
            if len(hist) >= 3:
                vel = (np.array(hist[-1]) - np.array(hist[0])) / (len(hist) - 1)
        if state == "none" and (n - last_tag_n) > (HOLD_S + PRED_S) * fps:
            staff_id = None; hist.clear(); vel = np.zeros(2)

        rows.append(dict(frame=n, time_s=round((n - 1) / fps, 2), staff_present=int(state != "none"),
                         state=state, track_id=staff_id if state != "none" else "",
                         cx=round(cx, 1) if cx is not None else "", cy=round(cy, 1) if cy is not None else "",
                         radius_px=round(radius), tag_conf=round(conf, 2)))

        # 4) draw: thin green box on every strong tag; GREEN dot + ring when sure (tag / track),
        #    YELLOW circle (radius grows with time since the last sure frame) when estimated
        img = frame; t = (n - 1) / fps; F = cv2.FONT_HERSHEY_SIMPLEX
        GREEN, YELLOW = (0, 255, 0), (0, 255, 255)
        for c, (x1, y1, x2, y2) in strong_tags:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(img, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), GREEN, 1)
            cv2.putText(img, f"tag {c:.2f}", (x1 - 8, y1 - 12), F, 0.4, GREEN, 1)
        if state != "none":
            solid = state in ("tag", "track", "tag-only")
            qx, qy = min(max(int(cx), 0), W - 1), min(max(int(cy), 0), H - 1)
            col = GREEN if solid else YELLOW
            if solid:
                cv2.circle(img, (qx, qy), 6, col, -1); cv2.circle(img, (qx, qy), 12, col, 2)
            else:
                cv2.circle(img, (qx, qy), int(radius), col, 1); cv2.circle(img, (qx, qy), 3, col, -1)
            cv2.putText(img, f"STAFF ({qx},{qy})" + ("" if solid else " ~"), (qx + 16, max(14, qy - 10)), F, 0.6, col, 2)
        cv2.rectangle(img, (0, 0), (360, 26), (0, 0, 0), -1)
        cv2.putText(img, f"t={t:5.2f}s  frame {n - 1}  staff: {state}", (5, 18), F, 0.5, (255, 255, 255), 1)
        writer.write(img)
        if n % 100 == 0:
            print(f"frame {n} ...")

    cap.release(); writer.release()

    # 5) results
    os.remove(yaml_path)
    d = pd.DataFrame(rows)
    hit = d[d.tag_conf >= TAG_STRONG]
    segs = []
    if len(hit):
        fr = hit.frame.tolist(); s = p = fr[0]
        for f in fr[1:] + [None]:
            if f is None or f - p > GAP_FRAMES:
                sub = hit[(hit.frame >= s) & (hit.frame <= p)]
                segs.append(dict(start_frame=s, end_frame=p, start_s=round((s - 1) / fps, 2), end_s=round((p - 1) / fps, 2),
                                 frames_with_tag=len(sub), best_conf=sub.tag_conf.max()))
                s = f
            p = f if f is not None else p
    seg = pd.DataFrame(segs)

    print(f"\nvideo: {len(d)} frames ({len(d) / fps:.1f} s)")
    print(f"frames with a clearly seen tag (>= {TAG_STRONG}): {len(hit)} ({100 * len(hit) / len(d):.1f}%) in {len(seg)} segments")
    print(f"frames with a staff position (incl. estimated): {int(d.staff_present.sum())}")
    print(d.state.value_counts().to_string())
    if len(seg):
        print(seg.to_string(index=False))
    print("video saved to:", out_video)


if __name__ == "__main__":
    main()