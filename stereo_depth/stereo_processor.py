"""
StereoProcessor
---------------
Encapsulates the stereo-vision pipeline:

Three rectification modes are supported:

  * "none"         - Skip rectification entirely. Use the raw images
                     directly. This is the right choice when the cameras
                     are already aligned (simulation environments like
                     Gazebo, or a properly calibrated rig publishing
                     pre-rectified images).
  * "uncalibrated" - Original project's method: SIFT + BFMatcher +
                     RANSAC 8-point -> F -> cv2.stereoRectifyUncalibrated
                     -> H1, H2. Computed once on the first frame.
                     Geometrically correct for epipolar matching but
                     can warp images significantly, especially when
                     the cameras are actually already aligned.
  * "calibrated"   - Use the CameraInfo's projection matrix P (which
                     ROS publishes pre-rectified) via cv2.initUndistort-
                     RectifyMap with K, D, R, P. Needs a properly
                     calibrated stereo pair.

Per-frame pipeline (after rectification, if any):
  - StereoSGBM disparity
  - depth_mm = (baseline_mm * f_px) / disparity_px
"""
import numpy as np
import cv2

from .utils.fundamental_matrix import get_inliers
from .utils.misc_utils import sift_features_to_array


VALID_MODES = ("none", "uncalibrated", "calibrated")


class StereoProcessor:
    def __init__(self, baseline_mm=80.0,
                 sgbm_num_disparities=128,
                 sgbm_block_size=5,
                 max_features=300,
                 rectification_mode="none"):
        self.baseline_mm = float(baseline_mm)
        self.max_features = int(max_features)

        if rectification_mode not in VALID_MODES:
            raise ValueError(
                f"rectification_mode must be one of {VALID_MODES}, "
                f"got '{rectification_mode}'")
        self.mode = rectification_mode

        # Rectification state (used by "uncalibrated" mode)
        self.H1 = None
        self.H2 = None
        self.F = None
        # Maps used by "calibrated" mode
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None
        # Generic "are we ready to rectify?" flag
        self.calibrated = (self.mode == "none")  # "none" is always ready

        # Camera intrinsics (filled from CameraInfo)
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.R1_rect = None   # CameraInfo.r (3x3) - left
        self.R2_rect = None   # CameraInfo.r (3x3) - right
        self.P1 = None        # CameraInfo.p (3x4) - left
        self.P2 = None        # CameraInfo.p (3x4) - right
        self.image_size = None
        self.focal_length = None  # average of fx1, fx2

        # SGBM
        if sgbm_num_disparities % 16 != 0:
            sgbm_num_disparities = ((sgbm_num_disparities // 16) + 1) * 16
        bs = sgbm_block_size
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=sgbm_num_disparities,
            blockSize=bs,
            P1=8 * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    # ------------------------------------------------------------------ #
    #  Camera info                                                       #
    # ------------------------------------------------------------------ #
    def set_camera_info(self, K1, K2,
                        D1=None, D2=None,
                        R1=None, R2=None,
                        P1=None, P2=None,
                        image_size=None):
        self.K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
        self.K2 = np.asarray(K2, dtype=np.float64).reshape(3, 3)
        if D1 is not None:
            self.D1 = np.asarray(D1, dtype=np.float64).ravel()
        if D2 is not None:
            self.D2 = np.asarray(D2, dtype=np.float64).ravel()
        if R1 is not None:
            self.R1_rect = np.asarray(R1, dtype=np.float64).reshape(3, 3)
        if R2 is not None:
            self.R2_rect = np.asarray(R2, dtype=np.float64).reshape(3, 3)
        if P1 is not None:
            self.P1 = np.asarray(P1, dtype=np.float64).reshape(3, 4)
        if P2 is not None:
            self.P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
        if image_size is not None:
            self.image_size = (int(image_size[0]), int(image_size[1]))  # (w,h)

        # Average of the two focal lengths (in pixels)
        # In calibrated mode prefer fx from P (the rectified focal length).
        if self.mode == "calibrated" and self.P1 is not None and self.P2 is not None:
            self.focal_length = 0.5 * (self.P1[0, 0] + self.P2[0, 0])
        else:
            self.focal_length = 0.5 * (self.K1[0, 0] + self.K2[0, 0])

        # For calibrated mode, try to build undistort/rectify maps now.
        if self.mode == "calibrated":
            self._build_calibrated_maps()

    # ------------------------------------------------------------------ #
    #  Calibration                                                       #
    # ------------------------------------------------------------------ #
    def _build_calibrated_maps(self):
        """Build cv2.initUndistortRectifyMap LUTs for calibrated mode."""
        if (self.K1 is None or self.K2 is None or
                self.R1_rect is None or self.R2_rect is None or
                self.P1 is None or self.P2 is None or
                self.image_size is None):
            return False
        D1 = self.D1 if self.D1 is not None else np.zeros(5)
        D2 = self.D2 if self.D2 is not None else np.zeros(5)
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.K1, D1, self.R1_rect, self.P1,
            self.image_size, cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.K2, D2, self.R2_rect, self.P2,
            self.image_size, cv2.CV_32FC1)
        self.calibrated = True
        return True

    def calibrate(self, img_left, img_right):
        """
        Prepare rectification for the current mode.
          - "none"         : trivially ready (no-op).
          - "calibrated"   : built from CameraInfo, only checks readiness.
          - "uncalibrated" : runs SIFT + RANSAC and builds H1, H2.
        Returns True on success.
        """
        if self.mode == "none":
            self.calibrated = True
            return True

        if self.mode == "calibrated":
            return self._build_calibrated_maps()

        # --- "uncalibrated" branch (original project's algorithm) ---
        if img_left is None or img_right is None:
            return False
        if img_left.shape[:2] != img_right.shape[:2]:
            return False

        gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) \
            if img_left.ndim == 3 else img_left
        gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) \
            if img_right.ndim == 3 else img_right

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_l, None)
        kp2, des2 = sift.detectAndCompute(gray_r, None)
        if des1 is None or des2 is None:
            return False
        if len(kp1) < 8 or len(kp2) < 8:
            return False

        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        chosen = matches[: self.max_features]
        if len(chosen) < 8:
            return False

        pairs = sift_features_to_array(chosen, kp1, kp2)

        F_best, inliers = get_inliers(pairs)
        if F_best is None or inliers.shape[0] < 8:
            return False

        set1 = inliers[:, 0:2].astype(np.float32)
        set2 = inliers[:, 2:4].astype(np.float32)

        h, w = gray_l.shape[:2]
        ok, H1, H2 = cv2.stereoRectifyUncalibrated(
            set1.reshape(-1, 1, 2),
            set2.reshape(-1, 1, 2),
            F_best,
            imgSize=(w, h),
        )
        if not ok:
            return False

        self.F = F_best
        self.H1 = H1
        self.H2 = H2
        self.calibrated = True
        return True

    def reset_calibration(self):
        if self.mode == "none":
            return  # nothing to reset
        self.calibrated = False
        self.H1 = self.H2 = self.F = None
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None

    # ------------------------------------------------------------------ #
    #  Per-frame pipeline                                                #
    # ------------------------------------------------------------------ #
    def rectify(self, img_left, img_right):
        if self.mode == "none":
            return img_left, img_right
        if not self.calibrated:
            return None, None
        if self.mode == "uncalibrated":
            h, w = img_left.shape[:2]
            rect_l = cv2.warpPerspective(img_left, self.H1, (w, h))
            rect_r = cv2.warpPerspective(img_right, self.H2, (w, h))
            return rect_l, rect_r
        # calibrated
        rect_l = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def compute_disparity(self, rect_left, rect_right):
        g_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY) \
            if rect_left.ndim == 3 else rect_left
        g_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY) \
            if rect_right.ndim == 3 else rect_right
        # SGBM returns disparity * 16 in int16
        disp = self.sgbm.compute(g_l, g_r).astype(np.float32) / 16.0
        return disp

    def compute_depth_mm(self, disparity):
        """depth[mm] = (baseline_mm * f_px) / disparity[px]."""
        if self.focal_length is None:
            return None
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.5  # ignore invalid / very-far pixels
        depth[valid] = (self.baseline_mm * self.focal_length) / disparity[valid]
        return depth

    @staticmethod
    def region_median_depth(depth_map, mask):
        """Median of valid (>0) depth values inside a binary mask."""
        if depth_map is None or mask is None:
            return None
        sel = depth_map[(mask > 0) & (depth_map > 0)]
        if sel.size == 0:
            return None
        return float(np.median(sel))