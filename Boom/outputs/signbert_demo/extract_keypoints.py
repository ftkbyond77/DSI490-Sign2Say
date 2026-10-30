"""MediaPipe-family ONNX models through OpenCV CPU; no GPU/OpenGL/MANO runtime."""
from pathlib import Path
import sys,argparse
import cv2
import numpy as np
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'vendor/opencv_zoo'))
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose
from mp_persondet import MPPersonDet
from mp_pose import MPPose
UPPER_POSE_IDX=[0,2,5,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
PIPELINE_VERSION='opencv-mediapipe-onnx-cpu-v1'
ASSETS=ROOT/'assets/onnx'

def extract_video_keypoints(video_path,target_fps=15):
    cv2.setNumThreads(4)
    options=dict(backendId=cv2.dnn.DNN_BACKEND_OPENCV,targetId=cv2.dnn.DNN_TARGET_CPU)
    palm_detector=MPPalmDet(str(ASSETS/'palm_detection_mediapipe_2023feb.onnx'),**options)
    hand_model=MPHandPose(str(ASSETS/'handpose_estimation_mediapipe_2023feb.onnx'),confThreshold=.5,**options)
    person_detector=MPPersonDet(str(ASSETS/'person_detection_mediapipe_2023mar.onnx'),**options)
    pose_model=MPPose(str(ASSETS/'pose_estimation_mediapipe_2023mar.onnx'),**options)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {video_path}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: fps=30.
    stride=max(1,round(fps/target_fps));frame_no=0
    hand_seq=[];pose_seq=[];timestamps=[];last_person=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            current=frame_no;frame_no+=1
            if current%stride: continue
            height,width=frame.shape[:2];scale=np.array([width,height,width])
            persons=person_detector.infer(frame)
            body_result=None
            if len(persons):
                person=persons[np.argmax(persons[:,-1])].copy()
                body_result=pose_model.infer(frame,person.copy())
                if body_result is not None: last_person=person
            if body_result is None and last_person is not None:
                body_result=pose_model.infer(frame,last_person.copy())
            pose=np.full((19,3),np.nan,dtype=np.float32)
            if body_result is not None:
                landmarks=body_result[1][UPPER_POSE_IDX]
                valid=(landmarks[:,3]>=.5)&(landmarks[:,4]>=.5)
                pose[valid]=landmarks[valid,:3]/scale
            detected=[]
            palms=palm_detector.infer(frame)
            for palm in sorted(palms,key=lambda p:-p[-1])[:2]:
                hand=hand_model.infer(frame,palm.copy())
                if hand is not None: detected.append((hand[4:67].reshape(21,3)/scale,float(hand[130]),float(hand[131])))
            hands=np.full((2,21,3),np.nan,dtype=np.float32)
            wrists=pose[[9,10],:2]
            if detected and np.isfinite(wrists).all():
                if len(detected)==2:
                    direct=sum(np.linalg.norm(detected[i][0][0,:2]-wrists[i]) for i in range(2))
                    swapped=sum(np.linalg.norm(detected[1-i][0][0,:2]-wrists[i]) for i in range(2))
                    for side,index in enumerate((0,1) if direct<=swapped else (1,0)): hands[side]=detected[index][0]
                else:
                    side=int(np.linalg.norm(wrists-detected[0][0][0,:2],axis=-1).argmin());hands[side]=detected[0][0]
            else:
                confidences=[-1.,-1.]
                for points,handedness,confidence in detected:
                    side=int(handedness>=.5)
                    if confidence>confidences[side]: hands[side]=points;confidences[side]=confidence
            hand_seq.append(hands);pose_seq.append(pose);timestamps.append(round(current/fps*1000))
    finally: cap.release()
    if not hand_seq: raise ValueError('No decoded frames')
    hand_array=np.stack(hand_seq)
    return dict(left_hand=hand_array[:,0],right_hand=hand_array[:,1],pose=np.stack(pose_seq),fps=fps/stride,timestamps_ms=np.array(timestamps),source_frames=frame_no)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('output');a=p.parse_args()
    raw=extract_video_keypoints(a.video);np.savez_compressed(a.output,**raw,pipeline_version=PIPELINE_VERSION)
    print('Saved',a.output,'frames',len(raw['pose']))
