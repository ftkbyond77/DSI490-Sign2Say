import mediapipe as mp
import cv2
from pathlib import Path
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
print('MediaPipe',mp.__version__,flush=True)
root=Path('outputs/signbert_demo/assets/tasks')
opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/'hand_landmarker.task'),delegate=BaseOptions.Delegate.CPU),num_hands=2)
with vision.HandLandmarker.create_from_options(opts) as detector:
    frame=cv2.imread('work/probe.png')
    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    result=detector.detect(image)
    print('Hands',len(result.hand_landmarks),flush=True)
opts=vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/'pose_landmarker.task'),delegate=BaseOptions.Delegate.CPU))
with vision.PoseLandmarker.create_from_options(opts) as detector:
    result=detector.detect(image)
    print('Pose',len(result.pose_landmarks),flush=True)
