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

Distance modes (what gets returned to the user):

  * "z"      - Cartesian Z component: perpendicular distance from the
               image plane. What a "depth map" classically means.
  * "radial" - Euclidean distance from the camera's optical centre to
               the 3D point: sqrt(X^2 + Y^2 + Z^2). More physically
               meaningful when the user points at off-centre pixels
               on a flat surface; the corner of a flat wall really IS
               further from the lens than its centre, and radial
               reflects that. Default for this build.

Per-frame pipeline (after rectification, if any):
  - StereoSGBM disparity
  - optional median post-filter to suppress per-pixel noise
  - depth_mm Z = (baseline_mm * f_px) / disparity_px
  - if distance_mode == "radial":  R(u,v) = Z * sqrt(1 + (u-cx)^2/fx^2
                                                       + (v-cy)^2/fy^2)
"""
import numpy as np
import cv2

from .utils.fundamental_matrix import get_inliers
from .utils.misc_utils import sift_features_to_array


VALID_MODES = ("none", "uncalibrated", "calibrated")
VALID_DISTANCE_MODES = ("z", "radial")


class StereoProcessor:
    def __init__(self, baseline_mm=80.0,
                 sgbm_num_disparities=128,
                 sgbm_block_size=5,
                 max_features=300,
                 rectification_mode="none",
                 distance_mode="radial",
                 median_filter_size=5):
        self.baseline_mm = float(baseline_mm)
        self.max_features = int(max_features)

        if rectification_mode not in VALID_MODES:
            raise ValueError(
                f"rectification_mode must be one of {VALID_MODES}, "
                f"got '{rectification_mode}'")
        self.mode = rectification_mode

        if distance_mode not in VALID_DISTANCE_MODES:
            raise ValueError(
                f"distance_mode must be one of {VALID_DISTANCE_MODES}, "
                f"got '{distance_mode}'")
        self.distance_mode = distance_mode

        # Median filter on disparity. Cheap noise suppression.
        # 0 or 1 -> disabled.
        mfs = int(median_filter_size)
        if mfs <= 1:
            mfs = 0
        elif mfs % 2 == 0:
            mfs += 1  # OpenCV needs odd kernel size
        self.median_filter_size = mfs

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
        self.R1_rect = None
        self.R2_rect = None
        self.P1 = None
        self.P2 = None
        self.image_size = None
        self.focal_length = None  # average of fx1, fx2

        # Cached radial scale factor map (lazy-built per image size).
        # radial = Z * radial_scale_map ;  shape = (h, w), dtype float32.
        self._radial_scale = None
        self._radial_scale_for_size = None  # (w, h) tuple

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
            self.image_size = (int(image_size[0]), int(image_size[1]))

        # Focal length: in calibrated mode use rectified fx from P.
        if self.mode == "calibrated" and self.P1 is not None and self.P2 is not None:
            self.focal_length = 0.5 * (self.P1[0, 0] + self.P2[0, 0])
        else:
            self.focal_length = 0.5 * (self.K1[0, 0] + self.K2[0, 0])

        if self.mode == "calibrated":
            self._build_calibrated_maps()

        # Camera intrinsics changed -> radial scale map needs rebuilding.
        self._radial_scale = None
        self._radial_scale_for_size = None

    # ------------------------------------------------------------------ #
    #  Calibration                                                       #
    # ------------------------------------------------------------------ #
    def _build_calibrated_maps(self):
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
        if self.mode == "none":
            self.calibrated = True
            return True

        if self.mode == "calibrated":
            return self._build_calibrated_maps()

        # "uncalibrated"
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
            return
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

        # Cheap median post-filter to suppress per-pixel speckles.
        # Operates only on valid pixels; invalid stays invalid.
        if self.median_filter_size >= 3:
            # cv2.medianBlur on float32 requires kernel <= 5 and CV_32F.
            disp = cv2.medianBlur(disp, self.median_filter_size)
        return disp

    # ------------------------------------------------------------------ #
    #  Depth / distance                                                  #
    # ------------------------------------------------------------------ #
    def _ensure_radial_scale(self, shape):
        """Build (and cache) a per-pixel scale map S(u,v) such that
        radial_distance = Z * S(u,v).

        Derivation: for a pinhole camera, a pixel (u,v) corresponds to
        a ray with direction (x', y', 1) in normalised coords, where
        x' = (u-cx)/fx and y' = (v-cy)/fy.  The 3D point at depth Z is
        (Z*x', Z*y', Z) and its Euclidean distance from the camera
        origin is Z * sqrt(1 + x'^2 + y'^2).  So S = sqrt(1 + x'^2 + y'^2).
        """
        h, w = shape[:2]
        if (self._radial_scale is not None and
                self._radial_scale_for_size == (w, h)):
            return self._radial_scale

        # Prefer rectified intrinsics from P when in calibrated mode.
        if self.mode == "calibrated" and self.P1 is not None:
            fx = float(self.P1[0, 0])
            fy = float(self.P1[1, 1])
            cx = float(self.P1[0, 2])
            cy = float(self.P1[1, 2])
        elif self.K1 is not None:
            fx = float(self.K1[0, 0])
            fy = float(self.K1[1, 1])
            cx = float(self.K1[0, 2])
            cy = float(self.K1[1, 2])
        else:
            # Fallback: assume principal point at image centre and a
            # crude focal length. Better than nothing if CameraInfo
            # was never received (shouldn't happen with our node).
            fx = fy = max(w, h)
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5

        # Build x', y' grids in normalised camera coords.
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        x_n = (uu - cx) / fx
        y_n = (vv - cy) / fy
        scale = np.sqrt(1.0 + x_n * x_n + y_n * y_n).astype(np.float32)

        self._radial_scale = scale
        self._radial_scale_for_size = (w, h)
        return scale

    def compute_depth_mm(self, disparity):
        """Returns a per-pixel distance map in millimetres.

        - distance_mode == "z":      Z = (baseline * f) / disparity
        - distance_mode == "radial": R = Z * sqrt(1 + x_n^2 + y_n^2)

        Invalid pixels (disparity too small) are returned as 0.
        """
        if self.focal_length is None:
            return None

        Z = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.5
        Z[valid] = (self.baseline_mm * self.focal_length) / disparity[valid]

        if self.distance_mode == "z":
            return Z

        # radial
        scale = self._ensure_radial_scale(disparity.shape)
        R = Z * scale  # invalid pixels stay 0 because Z is 0 there
        return R

    @staticmethod
    def region_median_depth(depth_map, mask):
        if depth_map is None or mask is None:
            return None
        sel = depth_map[(mask > 0) & (depth_map > 0)]
        if sel.size == 0:
            return None
        return float(np.median(sel))