"""
extract_keypoints.py
---------------------
สกัด keypoints (มือซ้าย, มือขวา, pose ท่อนบน) จากคลิปวิดีโอภาษามือ 1 ไฟล์
ด้วย MediaPipe Holistic แล้วบันทึกเป็น .npz

Usage:
    python3 extract_keypoints.py <video_path> <output_npz_path>

Output npz keys:
    left_hand   : (T, 21, 3)  -- 21 จุดต่อเฟรม, NaN ถ้าตรวจไม่เจอในเฟรมนั้น
    right_hand  : (T, 21, 3)
    pose        : (T, 25, 3)  -- ใช้เฉพาะท่อนบน (ไหล่ถึงข้อมือ+ใบหน้าบางจุด)
    n_frames    : int
    fps         : float
"""

import sys
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# ใช้เฉพาะ pose landmark ท่อนบนที่เกี่ยวกับภาษามือ (ไม่เอาขา/สะโพก)
# index อ้างอิงจาก mediapipe pose landmark model (33 จุดทั้งหมด)
UPPER_POSE_IDX = [
    0,  # nose
    2, 5,        # ตา ซ้าย-ขวา (ใช้ประมาณตำแหน่งหัว)
    7, 8,        # หู
    11, 12,      # ไหล่ ซ้าย-ขวา
    13, 14,      # ศอก ซ้าย-ขวา
    15, 16,      # ข้อมือ ซ้าย-ขวา
    17, 18, 19, 20, 21, 22,  # นิ้ว/ปลายมือคร่าวๆจาก pose (สำรอง เผื่อ hand model จับไม่ได้)
    23, 24,      # สะโพก (ใช้ช่วย normalize ระดับลำตัว)
]


def extract_video_keypoints(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"เปิดวิดีโอไม่ได้: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    left_hand_seq = []
    right_hand_seq = []
    pose_seq = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)

            # --- มือซ้าย ---
            if results.left_hand_landmarks:
                lh = np.array(
                    [[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                lh = np.full((21, 3), np.nan, dtype=np.float32)
            left_hand_seq.append(lh)

            # --- มือขวา ---
            if results.right_hand_landmarks:
                rh = np.array(
                    [[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                rh = np.full((21, 3), np.nan, dtype=np.float32)
            right_hand_seq.append(rh)

            # --- pose (เฉพาะท่อนบน) ---
            if results.pose_landmarks:
                all_pose = np.array(
                    [[p.x, p.y, p.z] for p in results.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                pose = all_pose[UPPER_POSE_IDX]
            else:
                pose = np.full((len(UPPER_POSE_IDX), 3), np.nan, dtype=np.float32)
            pose_seq.append(pose)

    cap.release()

    return {
        "left_hand": np.stack(left_hand_seq) if left_hand_seq else np.zeros((0, 21, 3)),
        "right_hand": np.stack(right_hand_seq) if right_hand_seq else np.zeros((0, 21, 3)),
        "pose": np.stack(pose_seq) if pose_seq else np.zeros((0, len(UPPER_POSE_IDX), 3)),
        "fps": fps,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 extract_keypoints.py <video_path> <output_npz_path>")
        sys.exit(1)

    video_path, out_path = sys.argv[1], sys.argv[2]
    result = extract_video_keypoints(video_path)

    n_frames = result["left_hand"].shape[0]
    detected_left = int(np.sum(~np.isnan(result["left_hand"][:, 0, 0])))
    detected_right = int(np.sum(~np.isnan(result["right_hand"][:, 0, 0])))

    np.savez_compressed(
        out_path,
        left_hand=result["left_hand"],
        right_hand=result["right_hand"],
        pose=result["pose"],
        fps=result["fps"],
        n_frames=n_frames,
    )

    print(f"[OK] {video_path}")
    print(f"     เฟรมทั้งหมด: {n_frames} | fps: {result['fps']:.1f}")
    print(f"     ตรวจเจอมือซ้าย: {detected_left}/{n_frames} เฟรม")
    print(f"     ตรวจเจอมือขวา: {detected_right}/{n_frames} เฟรม")
    print(f"     บันทึกที่: {out_path}")


if __name__ == "__main__":
    main()
