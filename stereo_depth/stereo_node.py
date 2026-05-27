"""
stereo_node
-----------
ROS2 Jazzy node that:

  * Subscribes (time-synchronised) to:
        /camera0/image_raw          (sensor_msgs/Image)   - left
        /camera1/image_raw          (sensor_msgs/Image)   - right
        /camera0/camera_info        (sensor_msgs/CameraInfo)
        /camera1/camera_info        (sensor_msgs/CameraInfo)
  * Runs SIFT + RANSAC once on the first usable frame to obtain
    uncalibrated rectification homographies (H1, H2).
  * For every subsequent frame:
        - Rectifies both images.
        - Computes a StereoSGBM disparity map.
        - Converts to depth: depth_mm = baseline_mm * f_px / disparity_px
  * Opens an OpenCV window.  The user can:
        - Left-click + drag  -> rectangular ROI
        - Right-click + drag -> circular ROI
    The median depth of the valid pixels inside the ROI is drawn on the
    image (in millimetres and metres).
  * Keys:
        k -> redo SIFT/RANSAC calibration on the next frame
        c -> clear the current selection
        q -> quit
"""
import sys

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

from stereo_depth.stereo_processor import StereoProcessor


WINDOW_MAIN = "Stereo Depth (left rectified)"
WINDOW_DISP = "Disparity (debug)"


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__("stereo_depth_node")

        # -------- Parameters --------
        self.declare_parameter("baseline_mm", 80.0)
        self.declare_parameter("left_image_topic", "/camera0/image_raw")
        self.declare_parameter("right_image_topic", "/camera1/image_raw")
        self.declare_parameter("left_info_topic", "/camera0/camera_info")
        self.declare_parameter("right_info_topic", "/camera1/camera_info")
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sgbm_num_disparities", 128)
        self.declare_parameter("sgbm_block_size", 5)
        self.declare_parameter("show_disparity_window", True)
        self.declare_parameter("rectification_mode", "none")

        baseline_mm = self.get_parameter("baseline_mm").value
        left_img = self.get_parameter("left_image_topic").value
        right_img = self.get_parameter("right_image_topic").value
        left_info = self.get_parameter("left_info_topic").value
        right_info = self.get_parameter("right_info_topic").value
        slop = float(self.get_parameter("sync_slop").value)
        num_disp = int(self.get_parameter("sgbm_num_disparities").value)
        block_size = int(self.get_parameter("sgbm_block_size").value)
        self.show_disp = bool(self.get_parameter("show_disparity_window").value)
        rect_mode = str(self.get_parameter("rectification_mode").value).lower()

        # -------- Stereo pipeline --------
        self.processor = StereoProcessor(
            baseline_mm=baseline_mm,
            sgbm_num_disparities=num_disp,
            sgbm_block_size=block_size,
            rectification_mode=rect_mode,
        )
        self.bridge = CvBridge()

        # -------- Camera info bookkeeping --------
        self._info_left = None   # dict with K, D, R, P, size
        self._info_right = None

        # -------- Mouse / selection state --------
        self.drawing = False
        self.shape_mode = None  # "rect" or "circle"
        self.start_pt = None
        self.end_pt = None
        self.locked_shape = None  # last completed selection dict

        # -------- Subscribers --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.left_img_sub = message_filters.Subscriber(self, Image, left_img, qos_profile=qos)
        self.right_img_sub = message_filters.Subscriber(self, Image, right_img, qos_profile=qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_img_sub, self.right_img_sub],
            queue_size=10,
            slop=slop,
        )
        self.sync.registerCallback(self.image_callback)

        self.create_subscription(CameraInfo, left_info, self.left_info_cb, qos)
        self.create_subscription(CameraInfo, right_info, self.right_info_cb, qos)

        # -------- OpenCV window --------
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_MAIN, self.on_mouse)
        if self.show_disp:
            cv2.namedWindow(WINDOW_DISP, cv2.WINDOW_NORMAL)

        # -------- Periodic GUI tick (so the window stays alive even
        #          before the first synced pair arrives)
        self.create_timer(0.03, self._gui_tick)

        self.get_logger().info(
            f"stereo_depth_node started.\n"
            f"  left  image: {left_img}\n"
            f"  right image: {right_img}\n"
            f"  left  info : {left_info}\n"
            f"  right info : {right_info}\n"
            f"  baseline   : {baseline_mm} mm\n"
            f"  rectif. mode: {rect_mode}\n"
            f"Controls:  L-drag = rectangle  |  R-drag = circle  |  "
            f"k = recalibrate  |  c = clear  |  q = quit"
        )

    # ---------------------------------------------------------------- #
    #  CameraInfo callbacks                                            #
    # ---------------------------------------------------------------- #
    @staticmethod
    def _info_to_dict(msg: CameraInfo):
        return {
            "K": np.array(msg.k, dtype=np.float64).reshape(3, 3),
            "D": np.array(msg.d, dtype=np.float64).ravel(),
            "R": np.array(msg.r, dtype=np.float64).reshape(3, 3),
            "P": np.array(msg.p, dtype=np.float64).reshape(3, 4),
            "size": (int(msg.width), int(msg.height)),
        }

    def left_info_cb(self, msg: CameraInfo):
        info = self._info_to_dict(msg)
        if self._info_left is None or not np.allclose(self._info_left["K"], info["K"]):
            self._info_left = info
            self._try_push_intrinsics()
        else:
            self._info_left = info  # silently refresh

    def right_info_cb(self, msg: CameraInfo):
        info = self._info_to_dict(msg)
        if self._info_right is None or not np.allclose(self._info_right["K"], info["K"]):
            self._info_right = info
            self._try_push_intrinsics()
        else:
            self._info_right = info

    def _try_push_intrinsics(self):
        if self._info_left is not None and self._info_right is not None:
            l, r = self._info_left, self._info_right
            self.processor.set_camera_info(
                K1=l["K"], K2=r["K"],
                D1=l["D"], D2=r["D"],
                R1=l["R"], R2=r["R"],
                P1=l["P"], P2=r["P"],
                image_size=l["size"],
            )
            self.get_logger().info(
                f"CameraInfo received.  mode={self.processor.mode}  "
                f"f={self.processor.focal_length:.2f} px  "
                f"size={l['size']}"
            )

    # ---------------------------------------------------------------- #
    #  Mouse handling                                                  #
    # ---------------------------------------------------------------- #
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.shape_mode = "rect"
            self.start_pt = (x, y)
            self.end_pt = (x, y)
            self.locked_shape = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.shape_mode = "circle"
            self.start_pt = (x, y)
            self.end_pt = (x, y)
            self.locked_shape = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.shape_mode == "rect":
            self.drawing = False
            self.end_pt = (x, y)
            self.locked_shape = {
                "type": "rect",
                "start": self.start_pt,
                "end": self.end_pt,
            }
            self.shape_mode = None

        elif event == cv2.EVENT_RBUTTONUP and self.shape_mode == "circle":
            self.drawing = False
            self.end_pt = (x, y)
            self.locked_shape = {
                "type": "circle",
                "start": self.start_pt,
                "end": self.end_pt,
            }
            self.shape_mode = None

    # ---------------------------------------------------------------- #
    #  Image callback                                                  #
    # ---------------------------------------------------------------- #
    def image_callback(self, left_msg: Image, right_msg: Image):
        if self._info_left is None or self._info_right is None:
            self.get_logger().warn(
                "CameraInfo not yet received - waiting...",
                throttle_duration_sec=2.0,
            )
            return

        try:
            left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
            right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Rectification setup. For mode "none" this is a no-op; for
        # "calibrated" it builds maps from CameraInfo; for "uncalibrated"
        # it runs SIFT + RANSAC on the first usable frame.
        if not self.processor.calibrated:
            self.get_logger().info(
                f"Preparing rectification (mode={self.processor.mode})..."
            )
            ok = self.processor.calibrate(left, right)
            if ok:
                self.get_logger().info("Rectification ready.")
            else:
                self.get_logger().warn(
                    "Rectification setup failed - will retry next frame."
                )
                cv2.imshow(WINDOW_MAIN, left)
                return

        rect_l, rect_r = self.processor.rectify(left, right)
        disparity = self.processor.compute_disparity(rect_l, rect_r)
        depth_mm = self.processor.compute_depth_mm(disparity)

        # ---- Draw ----
        display = rect_l.copy()
        shape_to_show = None
        if self.drawing and self.start_pt and self.end_pt:
            shape_to_show = {
                "type": self.shape_mode,
                "start": self.start_pt,
                "end": self.end_pt,
            }
        elif self.locked_shape is not None:
            shape_to_show = self.locked_shape

        if shape_to_show is not None:
            self._draw_selection_and_distance(display, depth_mm, shape_to_show)

        # Help text along the bottom
        h, w = display.shape[:2]
        cv2.putText(
            display,
            "L-drag: rect | R-drag: circle | k: recalib | c: clear | q: quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_MAIN, display)

        if self.show_disp:
            disp_vis = disparity.copy()
            disp_vis[disp_vis < 0] = 0
            if disp_vis.max() > 0:
                disp_vis = (disp_vis * 255.0 / disp_vis.max()).astype(np.uint8)
            else:
                disp_vis = disp_vis.astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            cv2.imshow(WINDOW_DISP, disp_color)

    # ---------------------------------------------------------------- #
    #  Helpers                                                         #
    # ---------------------------------------------------------------- #
    def _draw_selection_and_distance(self, display, depth_map, shape):
        h, w = display.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        sx, sy = shape["start"]
        ex, ey = shape["end"]
        sx = int(np.clip(sx, 0, w - 1))
        sy = int(np.clip(sy, 0, h - 1))
        ex = int(np.clip(ex, 0, w - 1))
        ey = int(np.clip(ey, 0, h - 1))

        if shape["type"] == "rect":
            x0, x1 = sorted([sx, ex])
            y0, y1 = sorted([sy, ey])
            if x1 - x0 < 2 or y1 - y0 < 2:
                return
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
            text_pos = (x0, max(20, y0 - 10))
        else:  # circle
            cx, cy = sx, sy
            r = int(np.hypot(ex - sx, ey - sy))
            if r < 2:
                return
            cv2.circle(mask, (cx, cy), r, 255, -1)
            cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
            text_pos = (max(0, cx - r), max(20, cy - r - 10))

        med = StereoProcessor.region_median_depth(depth_map, mask)
        if med is None or med <= 0 or not np.isfinite(med):
            label = "Mesafe: gecersiz (valid disparity yok)"
        else:
            label = f"Mesafe: {med:.0f} mm  ({med / 1000.0:.3f} m)"

        # Background box for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = text_pos
        cv2.rectangle(display,
                      (tx - 4, ty - th - 6),
                      (tx + tw + 4, ty + 4),
                      (0, 0, 0), -1)
        cv2.putText(display, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    def _gui_tick(self):
        """Pump OpenCV's event loop and handle key presses."""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        if key == ord("q"):
            self.get_logger().info("Quit requested by user.")
            rclpy.shutdown()
        elif key == ord("k"):
            self.processor.reset_calibration()
            self.get_logger().info("Recalibration requested - will redo on next frame.")
        elif key == ord("c"):
            self.locked_shape = None
            self.start_pt = self.end_pt = None
            self.drawing = False
            self.shape_mode = None

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)