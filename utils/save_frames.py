# capture_when_both_tags.py
import os, cv2, numpy as np, pyrealsense2 as rs
from pupil_apriltags import Detector

MASTER_ID, OTHER_ID = 3, 3
TAG_FAMILY = "tag36h11"
TARGET = 25  # save this many frames

os.makedirs("calib_views", exist_ok=True)
det = Detector(families=TAG_FAMILY, nthreads=4, refine_edges=True)

W,H,FPS = 640,480,30
p,cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, W,H, rs.format.bgr8, FPS)
profile = p.start(cfg)

count = 0
try:
    while count < TARGET:
        fs = p.wait_for_frames()
        c = fs.get_color_frame()
        img = np.asanyarray(c.get_data())  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = det.detect(gray, estimate_tag_pose=False)
        ids = {int(d.tag_id) for d in dets}
        ok = (MASTER_ID in ids) and (OTHER_ID in ids)
        msg = f"Seen: {sorted(list(ids))}  | saving: {ok}  ({count}/{TARGET})"
        cv2.putText(img, msg, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255 if ok else 0, 0 if ok else 255), 2)
        cv2.imshow("capture", img)
        if ok:
            path = f"calib_views/{count:03d}.png"
            cv2.imwrite(path, img); count += 1
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
finally:
    p.stop(); cv2.destroyAllWindows()
    print("Saved", count, "frames in calib_views/")
