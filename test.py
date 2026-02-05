#!/usr/bin/env python3
"""
Advanced MediaPipe Eye Tracking with Head Pose Compensation
Implements research-grade improvements:
- Head pose compensation (yaw/pitch/roll)
- Eye-axis-aligned iris normalization
- Per-eye calibration with weighted fusion
- Stability-based calibration sampling
- Ridge regression with head pose features
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os
from collections import deque
import screeninfo

# Download the face landmarker model if not present
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading face landmarker model...")
    import ssl
    import certifi
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL, context=ssl_context) as response:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.read())
    except Exception:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(MODEL_URL, context=ssl_context) as response:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.read())
    print(f"Model downloaded to {MODEL_PATH}")

# Initialize MediaPipe Face Landmarker with transformation matrices ENABLED
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,  # CRITICAL: Enable for head pose
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# MediaPipe Iris Landmark Indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Additional landmarks for better eye height estimation
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Eye contour for visualization
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def extract_head_pose_features(transformation_matrix):
    """
    Extract head pose features from MediaPipe's 4x4 transformation matrix.
    Uses rotation matrix elements directly instead of Euler angles to avoid
    convention mismatch issues. Returns normalized pose features.
    """
    if transformation_matrix is None:
        return 0.0, 0.0, 0.0
    
    # Extract 3x3 rotation matrix
    R = transformation_matrix[:3, :3]
    
    # Use rotation matrix elements directly as features
    # This avoids Euler angle convention issues and is more numerically stable
    yaw_feat = R[2, 0]    # Forward-back coupling (left-right head turn)
    pitch_feat = R[2, 1]  # Up-down coupling (head tilt up/down)
    roll_feat = R[0, 1]   # Side tilt coupling
    
    return yaw_feat, pitch_feat, roll_feat


class SmoothValue:
    """Exponential smoothing for single values."""
    def __init__(self, alpha=0.5):
        self.value = None
        self.alpha = alpha

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


class SmoothPoint:
    """Exponential smoothing for gaze coordinates with confidence weighting."""
    def __init__(self, alpha=0.5):
        self.x = None
        self.y = None
        self.alpha = alpha

    def update(self, new_x, new_y, confidence=1.0):
        """Update with optional confidence weighting (0-1)."""
        effective_alpha = self.alpha * confidence
        if self.x is None:
            self.x = new_x
            self.y = new_y
        else:
            self.x = effective_alpha * new_x + (1 - effective_alpha) * self.x
            self.y = effective_alpha * new_y + (1 - effective_alpha) * self.y
        return self.x, self.y
    
    def reset(self):
        self.x = None
        self.y = None


def get_raw_ear(landmarks, top_idx, bottom_idx, inner_idx, outer_idx):
    """
    Calculate raw Eye Aspect Ratio (EAR).
    Returns the raw ratio without normalization.
    """
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]
    
    vertical = abs(top.y - bottom.y)
    horizontal = np.sqrt((outer.x - inner.x)**2 + (outer.y - inner.y)**2)
    
    if horizontal < 0.001:
        return 0.0
    
    return vertical / horizontal


class EyeOpennessCalibrator:
    """
    Per-user eye openness calibration.
    Learns min/max EAR dynamically for accurate blink detection.
    """
    def __init__(self, calibration_frames=60):
        self.left_ear_history = deque(maxlen=calibration_frames)
        self.right_ear_history = deque(maxlen=calibration_frames)
        self.left_min = 0.1
        self.left_max = 0.35
        self.right_min = 0.1
        self.right_max = 0.35
        self.is_calibrated = False
    
    def update(self, left_ear, right_ear):
        """Update with new EAR values. Auto-calibrates after enough samples."""
        self.left_ear_history.append(left_ear)
        self.right_ear_history.append(right_ear)
        
        if len(self.left_ear_history) >= 30:
            # Use percentiles to avoid outliers
            left_arr = np.array(self.left_ear_history)
            right_arr = np.array(self.right_ear_history)
            
            # 5th percentile = likely closed, 95th = likely open
            self.left_min = np.percentile(left_arr, 5)
            self.left_max = np.percentile(left_arr, 95)
            self.right_min = np.percentile(right_arr, 5)
            self.right_max = np.percentile(right_arr, 95)
            
            self.is_calibrated = True
    
    def get_openness(self, left_ear, right_ear):
        """Get normalized openness (0-1) for both eyes."""
        left_range = self.left_max - self.left_min + 1e-6
        right_range = self.right_max - self.right_min + 1e-6
        
        left_open = np.clip((left_ear - self.left_min) / left_range, 0.0, 1.0)
        right_open = np.clip((right_ear - self.right_min) / right_range, 0.0, 1.0)
        
        return left_open, right_open
    
    def reset(self):
        self.left_ear_history.clear()
        self.right_ear_history.clear()
        self.is_calibrated = False


def get_iris_position_eye_aligned(landmarks, iris_idx, inner_idx, outer_idx, top_idx, bottom_idx):
    """
    Calculate iris position in EYE-ALIGNED coordinate system.
    This is rotation-invariant - head tilt doesn't affect measurements.
    
    Returns (rel_x, rel_y) in roughly -1 to 1 range.
    """
    iris = landmarks[iris_idx]
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    
    # Eye center
    eye_center_x = (inner.x + outer.x) / 2
    eye_center_y = (inner.y + outer.y) / 2
    
    # Eye horizontal axis vector (from inner to outer corner)
    eye_vec_x = outer.x - inner.x
    eye_vec_y = outer.y - inner.y
    eye_width = np.sqrt(eye_vec_x**2 + eye_vec_y**2)
    
    if eye_width < 0.001:
        return 0.0, 0.0, 0.0
    
    # Normalize to unit vector
    eye_unit_x = eye_vec_x / eye_width
    eye_unit_y = eye_vec_y / eye_width
    
    # Perpendicular vector (90° rotation for vertical axis)
    perp_x = -eye_unit_y
    perp_y = eye_unit_x
    
    # Eye height (for vertical normalization)
    eye_height = abs(top.y - bottom.y)
    if eye_height < 0.001:
        eye_height = eye_width * 0.3  # Fallback ratio
    
    # Vector from eye center to iris
    iris_dx = iris.x - eye_center_x
    iris_dy = iris.y - eye_center_y
    
    # Project onto eye-aligned axes (dot product)
    # This is the KEY fix - rotation invariant!
    rel_x = (iris_dx * eye_unit_x + iris_dy * eye_unit_y) / (eye_width * 0.15)
    rel_y = (iris_dx * perp_x + iris_dy * perp_y) / (eye_height * 0.3)
    
    # Scale to roughly -1 to 1
    rel_x = np.clip(rel_x * 2.0, -2.0, 2.0)
    rel_y = np.clip(rel_y * 2.0, -2.0, 2.0)
    
    return rel_x, rel_y, eye_width


def draw_eye_contours(frame, landmarks, w, h):
    """Draw eye contours for visualization."""
    for contour in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
        for i in range(len(contour) - 1):
            pt1 = landmarks[contour[i]]
            pt2 = landmarks[contour[i + 1]]
            cv2.line(frame, (int(pt1.x * w), int(pt1.y * h)), 
                     (int(pt2.x * w), int(pt2.y * h)), (0, 255, 255), 1)
        pt1 = landmarks[contour[-1]]
        pt2 = landmarks[contour[0]]
        cv2.line(frame, (int(pt1.x * w), int(pt1.y * h)), 
                 (int(pt2.x * w), int(pt2.y * h)), (0, 255, 255), 1)


def draw_iris_points(frame, landmarks, w, h):
    """Draw iris center points."""
    if len(landmarks) >= 478:
        left_iris = landmarks[LEFT_IRIS_CENTER]
        cv2.circle(frame, (int(left_iris.x * w), int(left_iris.y * h)), 4, (0, 255, 0), -1)
        right_iris = landmarks[RIGHT_IRIS_CENTER]
        cv2.circle(frame, (int(right_iris.x * w), int(right_iris.y * h)), 4, (0, 255, 0), -1)


class PerEyeCalibration:
    """
    Per-eye calibration with head pose compensation.
    Uses Ridge Regression with head pose features for robustness.
    """
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.is_calibrated = False
        
        # Per-eye data storage
        self.left_eye_data = []
        self.right_eye_data = []
        self.head_pose_data = []
        self.screen_points = []
        
        # Regression coefficients (separate for x and y, separate for each eye)
        self.left_coef_x = None
        self.left_coef_y = None
        self.right_coef_x = None
        self.right_coef_y = None
        
        # Ridge regression regularization
        self.ridge_lambda = 0.1
        
        # 9-point calibration grid
        margin = 100
        self.target_points = [
            (margin, margin),
            (screen_width // 2, margin),
            (screen_width - margin, margin),
            (margin, screen_height // 2),
            (screen_width // 2, screen_height // 2),
            (screen_width - margin, screen_height // 2),
            (margin, screen_height - margin),
            (screen_width // 2, screen_height - margin),
            (screen_width - margin, screen_height - margin),
        ]
        self.current_point_idx = 0
        
        # Stability-based sampling
        self.stability_buffer = deque(maxlen=15)
        self.stability_threshold = 0.02  # Variance threshold
        self.stable_duration_required = 0.3  # Seconds
        self.stable_start_time = None
        self.sample_captured = False
    
    def get_current_target(self):
        if self.current_point_idx < len(self.target_points):
            return self.target_points[self.current_point_idx]
        return None
    
    def check_stability(self, lx, ly, rx, ry, yaw, pitch, roll):
        """
        Check if gaze is stable enough to capture.
        Only uses eye coordinates for stability (not head pose which has different units).
        Returns (is_stable, stability_score, time_stable).
        """
        self.stability_buffer.append((lx, ly, rx, ry, yaw, pitch, roll))
        
        if len(self.stability_buffer) < 10:
            return False, 0.0, 0.0
        
        arr = np.array(self.stability_buffer)
        # Only compute variance on eye coordinates (first 4 columns)
        # Head pose has different units and would skew the threshold
        eye_arr = arr[:, :4]  # lx, ly, rx, ry
        variance = np.var(eye_arr, axis=0).mean()
        
        # Also check head pose is relatively stable (separate threshold)
        head_arr = arr[:, 4:]  # yaw, pitch, roll
        head_variance = np.var(head_arr, axis=0).mean()
        head_stable = head_variance < 0.01  # Rotation matrix elements are ~[-1,1]
        
        stability_score = 1.0 - min(variance / self.stability_threshold, 1.0)
        is_stable = variance < self.stability_threshold and head_stable
        
        if is_stable:
            if self.stable_start_time is None:
                self.stable_start_time = time.time()
            time_stable = time.time() - self.stable_start_time
        else:
            self.stable_start_time = None
            time_stable = 0.0
        
        return is_stable, stability_score, time_stable
    
    def try_auto_capture(self, lx, ly, rx, ry, yaw, pitch, roll):
        """
        Attempt automatic capture when gaze is stable.
        Returns True when a point is captured.
        """
        is_stable, stability, time_stable = self.check_stability(lx, ly, rx, ry, yaw, pitch, roll)
        
        if is_stable and time_stable >= self.stable_duration_required and not self.sample_captured:
            # Capture this point
            self.left_eye_data.append((lx, ly))
            self.right_eye_data.append((rx, ry))
            self.head_pose_data.append((yaw, pitch, roll))
            self.screen_points.append(self.target_points[self.current_point_idx])
            
            self.current_point_idx += 1
            self.sample_captured = True
            self.stability_buffer.clear()
            self.stable_start_time = None
            
            if self.current_point_idx >= len(self.target_points):
                self._compute_mapping()
                return True, True  # point captured, calibration complete
            return True, False  # point captured, not complete
        
        if not is_stable:
            self.sample_captured = False
        
        return False, False
    
    def manual_capture(self, lx, ly, rx, ry, yaw, pitch, roll):
        """Manual capture with spacebar."""
        # Average the stability buffer if available
        if len(self.stability_buffer) >= 5:
            arr = np.array(self.stability_buffer)
            avg = arr.mean(axis=0)
            lx, ly, rx, ry, yaw, pitch, roll = avg
        
        self.left_eye_data.append((lx, ly))
        self.right_eye_data.append((rx, ry))
        self.head_pose_data.append((yaw, pitch, roll))
        self.screen_points.append(self.target_points[self.current_point_idx])
        
        self.current_point_idx += 1
        self.stability_buffer.clear()
        self.stable_start_time = None
        
        if self.current_point_idx >= len(self.target_points):
            self._compute_mapping()
            return True
        return False
    
    def _build_features(self, eye_x, eye_y, yaw, pitch, roll):
        """
        Build feature vector with interaction terms.
        Interaction terms model oculomotor coupling directly.
        Features: [1, ex, ey, yaw, pitch, ex*yaw, ex*pitch, ey*yaw, ey*pitch]
        """
        return np.array([
            1,
            eye_x,
            eye_y,
            yaw,
            pitch,
            eye_x * yaw,    # Horizontal gaze + head yaw coupling
            eye_x * pitch,  # Horizontal gaze + head pitch coupling
            eye_y * yaw,    # Vertical gaze + head yaw coupling  
            eye_y * pitch,  # Vertical gaze + head pitch coupling
        ])
    
    def _ridge_regression(self, X, y):
        """Ridge regression with regularization."""
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # Don't regularize intercept
        return np.linalg.solve(X.T @ X + self.ridge_lambda * I, X.T @ y)
    
    def _compute_mapping(self):
        """Compute per-eye polynomial + head pose regression."""
        if len(self.left_eye_data) < 4:
            return
        
        screen_x = np.array([p[0] for p in self.screen_points])
        screen_y = np.array([p[1] for p in self.screen_points])
        
        # Build feature matrices for each eye
        n = len(self.left_eye_data)
        
        # Left eye features
        X_left = np.zeros((n, 9))  # Updated feature count
        for i in range(n):
            lx, ly = self.left_eye_data[i]
            yaw, pitch, roll = self.head_pose_data[i]
            X_left[i] = self._build_features(lx, ly, yaw, pitch, roll)
        
        # Right eye features
        X_right = np.zeros((n, 9))  # Updated feature count
        for i in range(n):
            rx, ry = self.right_eye_data[i]
            yaw, pitch, roll = self.head_pose_data[i]
            X_right[i] = self._build_features(rx, ry, yaw, pitch, roll)
        
        # Fit per-eye models
        self.left_coef_x = self._ridge_regression(X_left, screen_x)
        self.left_coef_y = self._ridge_regression(X_left, screen_y)
        self.right_coef_x = self._ridge_regression(X_right, screen_x)
        self.right_coef_y = self._ridge_regression(X_right, screen_y)
        
        # Learn eye dominance from calibration residuals
        self._learn_eye_dominance(X_left, X_right, screen_x, screen_y)
        
        # Learn affine correction for camera-screen geometry
        self._learn_affine_correction(screen_x, screen_y)
        
        self.is_calibrated = True
        print(f"Calibration computed with {n} points")
    
    def _learn_eye_dominance(self, X_left, X_right, screen_x, screen_y):
        """Learn eye dominance from calibration residuals."""
        # Predict with each eye
        left_pred_x = X_left @ self.left_coef_x
        left_pred_y = X_left @ self.left_coef_y
        right_pred_x = X_right @ self.right_coef_x
        right_pred_y = X_right @ self.right_coef_y
        
        # Compute per-eye errors
        left_error = np.mean((left_pred_x - screen_x)**2 + (left_pred_y - screen_y)**2)
        right_error = np.mean((right_pred_x - screen_x)**2 + (right_pred_y - screen_y)**2)
        
        # Eye with lower error gets higher weight
        total_error = left_error + right_error + 1e-6
        self.eye_dominance = right_error / total_error  # Higher = favor left eye
        self.eye_dominance = np.clip(self.eye_dominance, 0.3, 0.7)  # Don't go extreme
        print(f"Eye dominance: {self.eye_dominance:.2f} (0.5=balanced, >0.5=left dominant)")
    
    def _learn_affine_correction(self, screen_x, screen_y):
        """
        Learn 2D affine correction for camera-screen geometry mismatch.
        Compensates for camera offset and tilt relative to screen.
        """
        # Get predictions using current model
        preds_x = []
        preds_y = []
        for i in range(len(self.left_eye_data)):
            lx, ly = self.left_eye_data[i]
            rx, ry = self.right_eye_data[i]
            yaw, pitch, roll = self.head_pose_data[i]
            
            left_features = self._build_features(lx, ly, yaw, pitch, roll)
            right_features = self._build_features(rx, ry, yaw, pitch, roll)
            
            pred_x = 0.5 * (np.dot(left_features, self.left_coef_x) + 
                          np.dot(right_features, self.right_coef_x))
            pred_y = 0.5 * (np.dot(left_features, self.left_coef_y) + 
                          np.dot(right_features, self.right_coef_y))
            preds_x.append(pred_x)
            preds_y.append(pred_y)
        
        preds_x = np.array(preds_x)
        preds_y = np.array(preds_y)
        
        # Build affine feature matrix [x, y, 1]
        A = np.column_stack([preds_x, preds_y, np.ones(len(preds_x))])
        
        # Solve for affine transform: [x', y'] = A @ [x, y, 1]
        self.affine_x, _, _, _ = np.linalg.lstsq(A, screen_x, rcond=None)
        self.affine_y, _, _, _ = np.linalg.lstsq(A, screen_y, rcond=None)
        
        # Check improvement
        corrected_x = A @ self.affine_x
        corrected_y = A @ self.affine_y
        orig_error = np.mean((preds_x - screen_x)**2 + (preds_y - screen_y)**2)
        corr_error = np.mean((corrected_x - screen_x)**2 + (corrected_y - screen_y)**2)
        print(f"Affine correction: error {np.sqrt(orig_error):.1f} -> {np.sqrt(corr_error):.1f} px")
    
    def map_to_screen(self, lx, ly, rx, ry, yaw, pitch, roll, 
                      left_openness=1.0, right_openness=1.0):
        """
        Map eye positions to screen with per-eye weighted fusion.
        Uses eye openness to weight contributions.
        """
        if not self.is_calibrated:
            # Pre-calibration: simple linear mapping with head pose correction
            # Compensate for head yaw/pitch
            k_yaw = 0.3
            k_pitch = 0.3
            
            avg_x = (lx + rx) / 2 - k_yaw * yaw
            avg_y = (ly + ry) / 2 - k_pitch * pitch
            
            sx = int((avg_x + 1) / 2 * self.screen_width)
            sy = int((avg_y + 1) / 2 * self.screen_height)
            return (np.clip(sx, 0, self.screen_width - 1), 
                    np.clip(sy, 0, self.screen_height - 1))
        
        # Build features
        left_features = self._build_features(lx, ly, yaw, pitch, roll)
        right_features = self._build_features(rx, ry, yaw, pitch, roll)
        
        # Per-eye predictions
        left_pred_x = np.dot(left_features, self.left_coef_x)
        left_pred_y = np.dot(left_features, self.left_coef_y)
        right_pred_x = np.dot(right_features, self.right_coef_x)
        right_pred_y = np.dot(right_features, self.right_coef_y)
        
        # Weighted fusion based on eye openness, head yaw, and learned dominance
        # When looking left, right eye is more visible (lower weight for left)
        yaw_weight = np.clip(0.5 - yaw * 0.5, 0.1, 0.9)
        
        # Apply learned eye dominance
        dominance = getattr(self, 'eye_dominance', 0.5)
        
        # Combine openness, yaw, and dominance
        left_weight = left_openness * yaw_weight * dominance
        right_weight = right_openness * (1 - yaw_weight) * (1 - dominance)
        
        total_weight = left_weight + right_weight
        if total_weight < 0.01:
            total_weight = 1.0
        
        left_weight /= total_weight
        right_weight /= total_weight
        
        # Fused prediction
        screen_x = left_weight * left_pred_x + right_weight * right_pred_x
        screen_y = left_weight * left_pred_y + right_weight * right_pred_y
        
        # Apply affine correction for camera-screen geometry
        if hasattr(self, 'affine_x') and hasattr(self, 'affine_y'):
            features = np.array([screen_x, screen_y, 1])
            screen_x = np.dot(features, self.affine_x)
            screen_y = np.dot(features, self.affine_y)
        
        screen_x = np.clip(screen_x, 0, self.screen_width - 1)
        screen_y = np.clip(screen_y, 0, self.screen_height - 1)
        
        return (int(screen_x), int(screen_y))
    
    def reset(self):
        self.left_eye_data = []
        self.right_eye_data = []
        self.head_pose_data = []
        self.screen_points = []
        self.is_calibrated = False
        self.current_point_idx = 0
        self.stability_buffer.clear()
        self.stable_start_time = None
        self.sample_captured = False
    
    def progress(self):
        return int(100 * self.current_point_idx / len(self.target_points))


def main():
    # Get screen dimensions
    try:
        screen = screeninfo.get_monitors()[0]
        SCREEN_WIDTH = screen.width
        SCREEN_HEIGHT = screen.height
    except Exception:
        SCREEN_WIDTH = 1440
        SCREEN_HEIGHT = 900
        print(f"Could not detect screen, using default: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    
    print(f"Screen detected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    
    # Use the new per-eye calibration with head pose
    calibration = PerEyeCalibration(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 60)
    print("Advanced Eye Tracking with Head Pose Compensation")
    print("=" * 60)
    print(f"Camera: {actual_width}x{actual_height}")
    print(f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print("\nKey Improvements:")
    print("  - Head pose compensation (rotation matrix features)")
    print("  - Eye-axis-aligned iris normalization")
    print("  - Per-eye calibration with weighted fusion")
    print("  - Stability-based auto-capture")
    print("  - Per-user eye openness calibration")
    print("  - Eye dominance learning")
    print("  - Camera-screen geometry correction")
    print("\nControls:")
    print("  'c' - Start/Reset calibration")
    print("  'SPACE' - Manual calibration capture")
    print("  'a' - Toggle auto-capture mode")
    print("  'g' - Toggle gaze window")
    print("  's' - Toggle iris display")
    print("  'q' - Quit")
    print("\nCalibration will AUTO-CAPTURE when gaze is stable!")
    print("="  * 60)
    
    # State variables
    show_iris = True
    show_debug = True
    show_gaze_window = False
    calibrating = False
    auto_capture = True  # Auto-capture when stable
    frame_count = 0
    
    # Per-user eye openness calibrator
    eye_openness_calibrator = EyeOpennessCalibrator(calibration_frames=60)
    
    # Smoothers with confidence weighting
    left_smoother = SmoothPoint(alpha=0.4)
    right_smoother = SmoothPoint(alpha=0.4)
    screen_smoother = SmoothPoint(alpha=0.25)
    
    # Head pose smoothers
    yaw_smoother = SmoothValue(alpha=0.3)
    pitch_smoother = SmoothValue(alpha=0.3)
    roll_smoother = SmoothValue(alpha=0.3)
    
    # FPS tracking
    fps_history = deque(maxlen=30)
    prev_time = time.perf_counter()
    
    # Current values for calibration
    current_lx, current_ly = 0.0, 0.0
    current_rx, current_ry = 0.0, 0.0
    current_yaw, current_pitch, current_roll = 0.0, 0.0, 0.0
    current_left_open, current_right_open = 1.0, 1.0
    
    h, w = actual_height, actual_width
    gaze_window_name = "Gaze Position"
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # FPS calculation
            current_time = time.perf_counter()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # Mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker.detect(mp_image)
            
            if results.face_landmarks and len(results.face_landmarks) > 0:
                landmarks = results.face_landmarks[0]
                num_landmarks = len(landmarks)
                
                # === EXTRACT HEAD POSE ===
                yaw, pitch, roll = 0.0, 0.0, 0.0
                if results.facial_transformation_matrixes and len(results.facial_transformation_matrixes) > 0:
                    transform_matrix = results.facial_transformation_matrixes[0]
                    yaw, pitch, roll = extract_head_pose_features(transform_matrix)
                    # Smooth head pose features
                    yaw = yaw_smoother.update(yaw)
                    pitch = pitch_smoother.update(pitch)
                    roll = roll_smoother.update(roll)
                
                current_yaw, current_pitch, current_roll = yaw, pitch, roll
                
                # Draw eye contours
                draw_eye_contours(frame, landmarks, w, h)
                
                # Draw iris if enabled
                if show_iris:
                    draw_iris_points(frame, landmarks, w, h)
                
                # Check if we have iris landmarks (478 total)
                if num_landmarks < 478:
                    cv2.putText(frame, f"No iris landmarks ({num_landmarks}/478)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # === EYE OPENNESS with per-user calibration ===
                    left_ear = get_raw_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                           LEFT_EYE_INNER, LEFT_EYE_OUTER)
                    right_ear = get_raw_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                            RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
                    
                    # Update calibrator and get normalized openness
                    eye_openness_calibrator.update(left_ear, right_ear)
                    left_openness, right_openness = eye_openness_calibrator.get_openness(left_ear, right_ear)
                    current_left_open = left_openness
                    current_right_open = right_openness
                    
                    # Skip if both eyes nearly closed (blink)
                    if left_openness < 0.2 and right_openness < 0.2:
                        cv2.putText(frame, "BLINK", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        continue
                    
                    # === GET EYE-ALIGNED IRIS POSITIONS ===
                    lx, ly, _ = get_iris_position_eye_aligned(
                        landmarks, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER,
                        LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
                    rx, ry, _ = get_iris_position_eye_aligned(
                        landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
                    
                    # Smooth per-eye with confidence
                    lx, ly = left_smoother.update(lx, ly, left_openness)
                    rx, ry = right_smoother.update(rx, ry, right_openness)
                    
                    current_lx, current_ly = lx, ly
                    current_rx, current_ry = rx, ry
                    
                    # === MAP TO SCREEN ===
                    screen_x, screen_y = calibration.map_to_screen(
                        lx, ly, rx, ry, yaw, pitch, roll,
                        left_openness, right_openness)
                    screen_x, screen_y = screen_smoother.update(screen_x, screen_y)
                    screen_x, screen_y = int(screen_x), int(screen_y)
                    
                    # === DEBUG INFO ===
                    if show_debug:
                        # Raw eye positions
                        cv2.putText(frame, f"L:({lx:.2f},{ly:.2f}) R:({rx:.2f},{ry:.2f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                        # Head pose features (not angles - they're rotation matrix elements)
                        cv2.putText(frame, f"Head: Y:{yaw:.2f} P:{pitch:.2f} R:{roll:.2f}", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                        # Eye openness
                        ear_status = "calibrated" if eye_openness_calibrator.is_calibrated else "calibrating..."
                        cv2.putText(frame, f"Open: L:{left_openness:.2f} R:{right_openness:.2f} ({ear_status})", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                        # Screen position
                        cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", 
                                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        status = "CALIBRATED" if calibration.is_calibrated else "Press 'c' to calibrate"
                        color = (0, 255, 0) if calibration.is_calibrated else (0, 165, 255)
                        cv2.putText(frame, status, (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # === GAZE VISUALIZATION WINDOW ===
                    if show_gaze_window:
                        # Full screen gaze display
                        gaze_display = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                        gaze_display[:] = (30, 30, 30)
                        
                        # Grid
                        for i in range(0, SCREEN_WIDTH, 100):
                            cv2.line(gaze_display, (i, 0), (i, SCREEN_HEIGHT), (50, 50, 50), 1)
                        for i in range(0, SCREEN_HEIGHT, 100):
                            cv2.line(gaze_display, (0, i), (SCREEN_WIDTH, i), (50, 50, 50), 1)
                        
                        # Calibration points
                        if calibration.is_calibrated:
                            for pt in calibration.screen_points:
                                cv2.circle(gaze_display, (int(pt[0]), int(pt[1])), 
                                          8, (100, 100, 100), -1)
                        
                        # Current gaze point
                        gx = screen_x
                        gy = screen_y
                        cv2.circle(gaze_display, (gx, gy), 25, (0, 0, 255), -1)
                        cv2.circle(gaze_display, (gx, gy), 35, (0, 255, 255), 3)
                        cv2.line(gaze_display, (gx - 50, gy), (gx + 50, gy), (0, 255, 255), 2)
                        cv2.line(gaze_display, (gx, gy - 50), (gx, gy + 50), (0, 255, 255), 2)
                        
                        cv2.putText(gaze_display, f"Gaze: ({screen_x}, {screen_y})", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                        cv2.putText(gaze_display, "Press 'g' to close", (20, SCREEN_HEIGHT - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                        
                        # Create fullscreen window
                        cv2.namedWindow(gaze_window_name, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(gaze_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow(gaze_window_name, gaze_display)
                    
                    # === CALIBRATION MODE ===
                    if calibrating and not calibration.is_calibrated:
                        target = calibration.get_current_target()
                        if target:
                            # Check stability
                            is_stable, stability, time_stable = calibration.check_stability(
                                lx, ly, rx, ry, yaw, pitch, roll)
                            
                            # Auto-capture if enabled
                            if auto_capture:
                                captured, complete = calibration.try_auto_capture(
                                    lx, ly, rx, ry, yaw, pitch, roll)
                                if captured:
                                    print(f"Auto-captured point {calibration.current_point_idx}/9")
                                if complete:
                                    calibrating = False
                                    cv2.destroyWindow("Calibration")
                                    print("=== CALIBRATION COMPLETE! ===")
                                    continue
                            
                            # Display calibration status
                            cv2.putText(frame, f"CALIBRATING - Point {calibration.current_point_idx + 1}/9", 
                                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            stability_color = (0, 255, 0) if is_stable else (0, 165, 255)
                            cv2.putText(frame, f"Stability: {stability*100:.0f}% ({time_stable:.1f}s)", 
                                       (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
                            cv2.putText(frame, f"Progress: {calibration.progress()}%", 
                                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Calibration window - FULLSCREEN
                            calib_display = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                            target_pos = (int(target[0]), int(target[1]))
                            
                            # Target dot - changes color based on stability
                            target_color = (0, 255, 0) if is_stable else (0, 0, 255)
                            cv2.circle(calib_display, target_pos, 40, target_color, -1)
                            cv2.circle(calib_display, target_pos, 50, (255, 255, 255), 3)
                            cv2.circle(calib_display, target_pos, 8, (255, 255, 255), -1)
                            
                            # Stability progress ring
                            if time_stable > 0:
                                progress_angle = int(360 * min(time_stable / calibration.stable_duration_required, 1.0))
                                cv2.ellipse(calib_display, target_pos, (60, 60), -90, 
                                           0, progress_angle, (0, 255, 255), 5)
                            
                            cv2.putText(calib_display, f"Point {calibration.current_point_idx + 1}/9", 
                                       (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                            
                            mode_text = "AUTO-CAPTURE: Hold gaze steady on the dot" if auto_capture else "Press SPACE to capture"
                            cv2.putText(calib_display, mode_text, 
                                       (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                            
                            if is_stable:
                                cv2.putText(calib_display, f"STABLE - capturing in {calibration.stable_duration_required - time_stable:.1f}s", 
                                           (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            
                            cv2.putText(calib_display, f"Progress: {calibration.progress()}%", 
                                       (40, SCREEN_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                            
                            # Create fullscreen calibration window
                            cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
                            cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.imshow("Calibration", calib_display)
            
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Status bar
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            auto_status = "AUTO" if auto_capture else "MANUAL"
            cv2.putText(frame, f"c:calib a:{auto_status} g:gaze q:quit", (w - 280, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Eye Tracking', frame)
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_iris = not show_iris
            elif key == ord('d'):
                show_debug = not show_debug
            elif key == ord('g'):
                show_gaze_window = not show_gaze_window
                if not show_gaze_window:
                    cv2.destroyWindow(gaze_window_name)
            elif key == ord('a'):
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
            elif key == ord('c'):
                calibration.reset()
                left_smoother.reset()
                right_smoother.reset()
                screen_smoother.reset()
                yaw_smoother.reset()
                pitch_smoother.reset()
                roll_smoother.reset()
                calibrating = True
                print("\n=== CALIBRATION STARTED ===")
                print("Look at each red dot and hold your gaze steady.")
                print("Points will auto-capture when stable, or press SPACE for manual capture.")
            elif key == ord(' ') and calibrating:
                if calibration.manual_capture(current_lx, current_ly, current_rx, current_ry,
                                              current_yaw, current_pitch, current_roll):
                    calibrating = False
                    cv2.destroyWindow("Calibration")
                    print("=== CALIBRATION COMPLETE! ===")
                else:
                    print(f"Manual capture: Point {calibration.current_point_idx}/9")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        print(f"\nTotal frames: {frame_count}, Avg FPS: {np.mean(fps_history):.1f}")


if __name__ == "__main__":
    main()