# save_extrinsics.py
import json, pyrealsense2 as rs
W,H,FPS = 640,480,30
p, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = p.start(cfg)

depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color = profile.get_stream(rs.stream.color).as_video_stream_profile()
ex = depth.get_extrinsics_to(color)  # maps Depth -> Colour
p.stop()

R = [ex.rotation[0:3], ex.rotation[3:6], ex.rotation[6:9]]
t = list(ex.translation)  # metres
json.dump({"R_dc": R, "t_dc_m": t}, open("extrinsics_depth_to_color.json","w"), indent=2)
print("R_dc:", R, "\nt_dc (m):", t)
