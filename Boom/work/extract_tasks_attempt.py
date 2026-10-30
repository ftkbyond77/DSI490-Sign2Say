"""CPU MediaPipe Tasks hand+pose extraction; no OpenGL or MANO required by this API."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

UPPER_POSE_IDX=[0,2,5,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
ASSETS=Path(__file__).parent/'assets/tasks'
PIPELINE_VERSION='tasks-cpu-v1'

def assign_hands(result,pose):
    output=np.full((2,21,3),np.nan,dtype=np.float32)
    detected=[np.array([[p.x,p.y,p.z] for p in landmarks],dtype=np.float32) for landmarks in result.hand_landmarks]
    # Anatomical left/right follows pose wrists, avoiding mirrored-video label conventions.
    wrists=pose[[9,10],:2]
    if detected and np.isfinite(wrists).all():
        if len(detected)==2:
            direct=np.linalg.norm(detected[0][0,:2]-wrists[0])+np.linalg.norm(detected[1][0,:2]-wrists[1])
            swapped=np.linalg.norm(detected[0][0,:2]-wrists[1])+np.linalg.norm(detected[1][0,:2]-wrists[0])
            order=(0,1) if direct<=swapped else (1,0)
            for side,index in enumerate(order): output[side]=detected[index]
        else:
            side=int(np.linalg.norm(wrists-detected[0][0,:2],axis=-1).argmin())
            output[side]=detected[0]
    else:
        confidence=[-1.,-1.]
        for points,categories in zip(detected,result.handedness):
            category=categories[0]; side=0 if category.category_name=='Left' else 1
            if category.score>confidence[side]: output[side]=points;confidence[side]=category.score
    return output

def extract_video_keypoints(video_path,target_fps=25):
    for name in ['hand_landmarker.task','pose_landmarker.task']:
        if not (ASSETS/name).is_file(): raise FileNotFoundError(f'Missing MediaPipe task model: {ASSETS/name}')
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {video_path}')
    source_fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps<=0: source_fps=30.
    stride=max(1,round(source_fps/target_fps))
    base=lambda name:BaseOptions(model_asset_path=str(ASSETS/name),delegate=BaseOptions.Delegate.CPU)
    hand_options=vision.HandLandmarkerOptions(base_options=base('hand_landmarker.task'),running_mode=vision.RunningMode.VIDEO,num_hands=2)
    pose_options=vision.PoseLandmarkerOptions(base_options=base('pose_landmarker.task'),running_mode=vision.RunningMode.VIDEO,num_poses=1)
    hands_seq=[];pose_seq=[];timestamps=[];frame_no=0;previous=-1
    try:
        with vision.HandLandmarker.create_from_options(hand_options) as hands, vision.PoseLandmarker.create_from_options(pose_options) as body:
            while True:
                ok,frame=cap.read()
                if not ok: break
                current=frame_no;frame_no+=1
                if current%stride: continue
                timestamp=max(previous+1,round(current/source_fps*1000));previous=timestamp
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                pose_result=body.detect_for_video(image,timestamp)
                pose=np.full((19,3),np.nan,dtype=np.float32)
                if pose_result.pose_landmarks:
                    landmarks=pose_result.pose_landmarks[0]
                    for j,index in enumerate(UPPER_POSE_IDX):
                        point=landmarks[index]
                        if point.visibility>=0.5 and point.presence>=0.5:
                            pose[j]=[point.x,point.y,point.z]
                result=hands.detect_for_video(image,timestamp)
                hands_seq.append(assign_hands(result,pose));pose_seq.append(pose);timestamps.append(timestamp)
    finally: cap.release()
    if not hands_seq: raise ValueError('No video frames could be decoded')
    hands_array=np.stack(hands_seq)
    return dict(left_hand=hands_array[:,0],right_hand=hands_array[:,1],pose=np.stack(pose_seq),fps=source_fps/stride,timestamps_ms=np.array(timestamps),source_frames=frame_no)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('output');a=p.parse_args()
    raw=extract_video_keypoints(a.video);np.savez_compressed(a.output,**raw,pipeline_version=PIPELINE_VERSION)
    print('Saved',a.output,'frames',len(raw['pose']))
