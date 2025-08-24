# save_intrinsics_auto.py
import json, pyrealsense2 as rs

p = rs.pipeline()
profile = p.start()  # no config -> SDK picks a valid colour+depth profile
try:
    sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = sp.get_intrinsics()
    K = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
         "width": intr.width, "height": intr.height}
    json.dump(K, open("../intrinsics_color.json", "w"), indent=2)
    print(K)
finally:
    p.stop()
