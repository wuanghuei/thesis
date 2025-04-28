import cv2
import numpy as np


def extract_pose_features(frame, pose_detector):
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return np.zeros(33 * 3)

    pose_landmarks = []
    for landmark in results.pose_landmarks.landmark:
        pose_landmarks.extend([landmark.x, landmark.y, landmark.z])

    return np.array(pose_landmarks)


def compute_velocity(pose_data):
    T, D = pose_data.shape
    velocity = np.zeros_like(pose_data)

    if T > 1:
        velocity[1:] = pose_data[1:] - pose_data[:-1]

    max_abs_val = np.max(np.abs(velocity)) + 1e-10
    velocity = velocity / max_abs_val

    return velocity
