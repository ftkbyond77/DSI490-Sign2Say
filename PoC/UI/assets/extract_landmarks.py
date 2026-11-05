"""Extract 21-pt hand landmarks per frame from the demo clips -> clip_*.json (normalized 0..1)."""
import json, sys, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
here = __file__.rsplit("assets",1)[0]+"assets/"
model = "D:/SLM-labs/SLM-Labs-SignDINO/webapp/web/public/models/hand_landmarker.task"
for name in (sys.argv[1:] or ["clip_a","clip_b"]):
    opts = vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=.4, min_tracking_confidence=.4)
    lm = vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(here+name+".mp4"); fps = cap.get(cv2.CAP_PROP_FPS); frames=[]; i=0
    while True:
        ok, fr = cap.read()
        if not ok: break
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        r = lm.detect_for_video(img, int(i*1000/fps))
        frames.append([[[round(p.x,4),round(p.y,4)] for p in h] for h in r.hand_landmarks]); i+=1
    json.dump({"fps":fps,"frames":frames}, open(here+name+".json","w"))
    det=[k for k,f in enumerate(frames) if f]
    print(name, len(frames), "frames; hands in", det[0]/fps if det else None, "->", det[-1]/fps if det else None, "n", len(det))
