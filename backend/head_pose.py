"""
head_pose.py – MediaPipe Face Mesh head pose estimation
Returns yaw, pitch, roll and classifies into 5 angles for registration.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# 3D model points of facial landmarks (generic face model)
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),      # Nose tip [1]
    (0.0,   -330.0, -65.0),     # Chin [152]
    (-225.0, 170.0, -135.0),    # Left eye left corner [263]
    (225.0,  170.0, -135.0),    # Right eye right corner [33]
    (-150.0,-150.0, -125.0),    # Left Mouth corner [287]
    (150.0, -150.0, -125.0),    # Right mouth corner [57]
], dtype=np.float64)

_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Angle thresholds (degrees)
_YAW_THRESH   = 20
_PITCH_THRESH = 15

ANGLES = ['front', 'left', 'right', 'up', 'down']

ANGLE_INSTRUCTIONS = {
    'front': 'Look straight at the camera',
    'left':  'Turn your head to the left',
    'right': 'Turn your head to the right',
    'up':    'Tilt your head up',
    'down':  'Tilt your head down',
}


class HeadPoseEstimator:
    def __init__(self):
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Returns dict with yaw, pitch, roll (degrees) and classified angle,
        or None if no face detected.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lms = result.multi_face_landmarks[0].landmark

        # 2D image points
        img_pts = np.array([
            (lms[i].x * w, lms[i].y * h) for i in _LANDMARK_IDS
        ], dtype=np.float64)

        # Camera internals
        focal = w
        cam_matrix = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS, img_pts, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles  # degrees

        angle_class = self._classify(yaw, pitch)

        return {
            "yaw":   round(float(yaw),   1),
            "pitch": round(float(pitch), 1),
            "roll":  round(float(roll),  1),
            "angle": angle_class,   # str or None
        }

    def _classify(self, yaw: float, pitch: float) -> str | None:
        if abs(yaw) < _YAW_THRESH and abs(pitch) < _PITCH_THRESH:
            return 'front'
        if yaw < -_YAW_THRESH:
            return 'left'
        if yaw > _YAW_THRESH:
            return 'right'
        if pitch > _PITCH_THRESH:
            return 'down'
        if pitch < -_PITCH_THRESH:
            return 'up'
        return None

    def close(self):
        self._mesh.close()
