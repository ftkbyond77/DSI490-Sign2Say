import sys,time
from pathlib import Path
import cv2
root=Path('outputs/signbert_demo');sys.path.insert(0,str(root/'vendor/opencv_zoo'))
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose
from mp_persondet import MPPersonDet
from mp_pose import MPPose
cv2.setNumThreads(4)
a=root/'assets/onnx';image=cv2.imread('work/probe.png')
t=time.time()
palms=MPPalmDet(str(a/'palm_detection_mediapipe_2023feb.onnx')).infer(image)
print('palms',palms.shape,flush=True)
hands=MPHandPose(str(a/'handpose_estimation_mediapipe_2023feb.onnx'),confThreshold=.5)
for palm in palms[:2]:
 hand=hands.infer(image,palm.copy());print('hand',None if hand is None else hand[-1],flush=True)
persons=MPPersonDet(str(a/'person_detection_mediapipe_2023mar.onnx')).infer(image)
print('persons',persons.shape,flush=True)
pose=MPPose(str(a/'pose_estimation_mediapipe_2023mar.onnx'))
for person in persons[:1]:
 result=pose.infer(image,person.copy());print('pose',None if result is None else result[-1],flush=True)
print('elapsed',time.time()-t)
