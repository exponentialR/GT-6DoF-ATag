# calibrate_intertag.py
import json, glob, numpy as np, cv2
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

MASTER_ID = 5   # change
OTHER_ID  = 4   # change
TAG_SIZE_M = 0.080  # metres (both tags identical size)
K = json.load(open("intrinsics_color.json"))
Kmat = np.array([[K["fx"],0,K["cx"]],[0,K["fy"],K["cy"]],[0,0,1]], float)

DET = Detector(families="tag36h11", nthreads=4, refine_edges=True)

def pnp(corners, size_m):
    s = size_m/2.0
    obj = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], np.float32)
    img = corners.astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, Kmat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    R_ct,_ = cv2.Rodrigues(rvec)
    return R_ct, tvec.reshape(3)

def inv(R_ab, t_ab):
    R_ba = R_ab.T
    t_ba = -(R_ab.T @ t_ab)
    return R_ba, t_ba

Rs, ts = [], []
for path in sorted(glob.glob("calib_views/*.png")):  # put your ~20 images here
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dets = DET.detect(img, estimate_tag_pose=False)
    dM = next((d for d in dets if int(d.tag_id)==MASTER_ID), None)
    dO = next((d for d in dets if int(d.tag_id)==OTHER_ID),  None)
    if not (dM and dO):
        continue
    R_cM, t_cM = pnp(dM.corners, TAG_SIZE_M)
    R_cO, t_cO = pnp(dO.corners, TAG_SIZE_M)
    # T_{O<-M} = (T_{C<-O})^{-1} * T_{C<-M}
    R_Oc, t_Oc = inv(R_cO, t_cO)
    R_OM = R_Oc @ R_cM
    t_OM = R_Oc @ t_cM + t_Oc
    Rs.append(R_OM); ts.append(t_OM)

if not Rs:
    raise SystemExit("No paired detections found.")

# robust average: quaternion mean for R, median for t
qs = R.from_matrix(np.stack(Rs,0)).as_quat()  # (N,4) xyzw
Q = (qs / np.linalg.norm(qs,axis=1,keepdims=True)).mean(0)
Q = Q / np.linalg.norm(Q)
R_OM_mean = R.from_quat(Q).as_matrix()
t_OM_med  = np.median(np.stack(ts,0), axis=0)

print("T_tag(other)<-tag(master):")
print("R =\n", R_OM_mean)
print("t (m) =", t_OM_med.tolist())

# Write back into tag_metadata.json for OTHER_ID as T_tag_from_obj (since object frame = master)
md = json.load(open("tag_metadata.json"))
for m in md:
    if int(m["tag_id"]) == OTHER_ID:
        m["T_tag_from_obj"]["R"] = R_OM_mean.tolist()
        m["T_tag_from_obj"]["t_m"] = t_OM_med.tolist()
json.dump(md, open("tag_metadata.json","w"), indent=2)
print("Updated tag_metadata.json for tag", OTHER_ID)
