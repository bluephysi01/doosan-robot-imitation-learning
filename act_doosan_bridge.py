#!/usr/bin/env python3
"""
Bridge an ACT policy checkpoint trained with LeRobot to a real Doosan robot via ROS2.

This script:
- Loads a local ACT checkpoint (pretrained_model directory from lerobot_train).
- Grabs images from two cameras (top and wrist) using OpenCV.
- Builds an observation matching the policy's expected features.
- Runs the ACT policy with the saved pre/post processors.
- Publishes a fake SO-101 JointState on /joint_states so that
  so_to_real_dsr_teleop.py can drive the real Doosan E0509.

Example:
python act_doosan_bridge.py \
    --pretrained_path outputs/train/.../pretrained_model \
    --device cuda \
    --rate 10 \
    --top_camera_index 0 \
    --wrist_camera_index 1 \
    --max_steps 800

In a separate terminal, run:
    ros2 run dsr_example2 so_to_real_dsr_teleop.py
"""

from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import JointState

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------

JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

# Doosan E0509 limits (rad)
DOOSAN_JOINT_LIMITS = {
    "Rotation": (-6.2832, 6.2832),
    "Pitch": (-6.2832, 6.2832),
    "Elbow": (-2.7053, 2.7053),
    "Wrist_Pitch": (-6.2832, 6.2832),
    "Wrist_Roll": (-6.2832, 6.2832),
    "Jaw": (0.0, 2.2),
}

# ------------------------------------------------------------------------------------
# CONVERSION HELPERS
# ------------------------------------------------------------------------------------

def convert_so101_to_radians(action_normalized: np.ndarray) -> np.ndarray:
    """Map [-100,100] → radians and clip to Doosan limits."""
    action_radians = action_normalized * (math.pi / 100.0)
    for i, name in enumerate(JOINT_NAMES):
        if name in DOOSAN_JOINT_LIMITS:
            lo, hi = DOOSAN_JOINT_LIMITS[name]
            action_radians[i] = np.clip(action_radians[i], lo, hi)
    return action_radians


def convert_doosan_to_so101_normalized(doosan_joints_deg: np.ndarray) -> np.ndarray:
    """Inverse mapping of so_to_real_dsr_teleop.py."""
    rad = np.radians(doosan_joints_deg)

    rot_rad = -(rad[0] + math.pi / 2)
    pitch_rad = -rad[1]
    elbow_rad = -(rad[2] + math.pi / 2)
    w_pitch_rad = -rad[4]
    w_roll_rad = rad[5]
    jaw_rad = 0.0

    return np.array(
        [
            rot_rad / (math.pi / 100),
            pitch_rad / (math.pi / 100),
            elbow_rad / (math.pi / 100),
            w_pitch_rad / (math.pi / 100),
            w_roll_rad / (math.pi / 100),
            jaw_rad / (math.pi / 100),
        ],
        dtype=np.float32,
    )

# ------------------------------------------------------------------------------------
# MAIN ROS NODE
# ------------------------------------------------------------------------------------

class ACTDoosanBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("act_doosan_bridge")

        self.args = args
        self.device = get_safe_torch_device(args.device, log=True)
        self.display_data = bool(args.display_data)
        self._display_error_logged = False

        # ----------------------------
        # Load ACT policy
        # ----------------------------
        policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
        policy_cfg.device = self.device.type
        if args.use_amp is not None:
            policy_cfg.use_amp = args.use_amp
        policy_cfg.pretrained_path = args.pretrained_path

        self.policy = ACTPolicy.from_pretrained(
            args.pretrained_path, config=policy_cfg
        )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )

        # ----------------------------
        # Publishers / Subscribers
        # ----------------------------
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.robot_id = args.robot_id
        self.doosan_joint_sub = self.create_subscription(
            JointState,
            f"/{self.robot_id}/joint_states",
            self._doosan_joint_callback,
            qos,
        )

        self.current_doosan_joints_deg = None
        self.joint_state_received = False

        # ----------------------------
        # Timing
        # ----------------------------
        self.rate_hz = float(args.rate)
        self.dt = 1.0 / max(self.rate_hz, 1e-3)
        self.max_steps = args.max_steps

        # ----------------------------
        # Smoothing / scaling
        # ----------------------------
        self.smoothing_alpha = args.smoothing_alpha
        self.action_scale = args.action_scale
        self.smoothed_action = None
        self.step_counter = 0

        # ----------------------------
        # OpenCV camera init
        # ----------------------------
        self.cap_top = self._open_camera(
            args.top_camera_index, args.camera_width, args.camera_height, "top"
        )
        self.cap_wrist = self._open_camera(
            args.wrist_camera_index, args.camera_width, args.camera_height, "wrist"
        )

        # ----------------------------
        # Visualization (Rerun)
        # ----------------------------
        if self.display_data:
            try:
                init_rerun(session_name="act_doosan_inference")
            except Exception as e:
                self.display_data = False
                self.get_logger().error(f"Failed to init Rerun: {e}")

        self.get_logger().info(f"ACT checkpoint: {args.pretrained_path}")
        self.get_logger().info(f"Device: {self.device}")

    # ----------------------------------------------------------------------
    # Camera helpers
    # ----------------------------------------------------------------------

    def _open_camera(self, index, w, h, name):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {name} camera index={index}")
        if w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    def _read_camera(self, cap, w, h, name):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read {name} camera")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (w > 0 and h > 0) and (frame.shape[1] != w or frame.shape[0] != h):
            frame = cv2.resize(frame, (w, h))
        return frame.astype(np.uint8)

    # ----------------------------------------------------------------------
    # Observation builder
    # ----------------------------------------------------------------------

    def _build_observation(self) -> dict[str, np.ndarray]:
        top_img = self._read_camera(
            self.cap_top, self.args.camera_width, self.args.camera_height, "top"
        )
        wrist_img = self._read_camera(
            self.cap_wrist, self.args.camera_width, self.args.camera_height, "wrist"
        )

        if self.current_doosan_joints_deg is not None:
            state = convert_doosan_to_so101_normalized(
                self.current_doosan_joints_deg
            )
        else:
            state = np.zeros(6, dtype=np.float32)

        return {
            "observation.state": state,
            "observation.images.top": top_img,
            "observation.images.wrist": wrist_img,
        }

    # ----------------------------------------------------------------------
    # Rerun logging helper
    # ----------------------------------------------------------------------

    def _log_visualization(
        self,
        observation: dict[str, np.ndarray],
        action_normalized: np.ndarray,
        action_scaled: np.ndarray,
        action_radians: np.ndarray,
    ) -> None:
        if not self.display_data:
            return

        # lerobot_record 와 비슷한 키 구조(`observation.top`, `observation.wrist` 등)를
        # 사용하기 위해, 시각화용으로만 키를 재매핑한다.
        obs_log: dict[str, np.ndarray] = {}
        if "observation.images.top" in observation:
            obs_log["top"] = observation["observation.images.top"]
        if "observation.images.wrist" in observation:
            obs_log["wrist"] = observation["observation.images.wrist"]
        if "observation.state" in observation:
            obs_log["state"] = observation["observation.state"]
        if self.current_doosan_joints_deg is not None:
            obs_log["doosan_degrees"] = self.current_doosan_joints_deg.astype(
                np.float32
            )

        action_log = {
            "normalized": action_normalized,
            "scaled": action_scaled,
            "radians": action_radians,
        }
        if self.smoothed_action is not None:
            action_log["smoothed"] = self.smoothed_action

        try:
            log_rerun_data(observation=obs_log, action=action_log)
        except Exception as exc:
            if not self._display_error_logged:
                self.get_logger().warn(
                    f"Rerun logging failed, disabling display: {exc}"
                )
                self._display_error_logged = True
            self.display_data = False

    # ----------------------------------------------------------------------
    # Doosan joint state callback
    # ----------------------------------------------------------------------

    def _doosan_joint_callback(self, msg: JointState) -> None:
        if len(msg.position) < 6:
            return

        joints = dict(zip(msg.name, msg.position))
        try:
            rad = np.array(
                [
                    joints["joint_1"],
                    joints["joint_2"],
                    joints["joint_3"],
                    joints["joint_4"],
                    joints["joint_5"],
                    joints["joint_6"],
                ],
                dtype=np.float32,
            )
        except KeyError:
            return

        self.current_doosan_joints_deg = np.degrees(rad)

        if not self.joint_state_received:
            self.joint_state_received = True
            self.get_logger().info("✓ Doosan joint state received")

    # ----------------------------------------------------------------------
    # Publish fake SO-101 joint state
    # ----------------------------------------------------------------------

    def _publish_joint_state(self, action: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in action]
        self.joint_pub.publish(msg)

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def spin(self) -> None:
        self.get_logger().info("Waiting for Doosan joint states...")

        # Wait up to 5 seconds
        for _ in range(50):
            if self.joint_state_received:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.joint_state_received:
            self.get_logger().error("No Doosan joint states received!")
        else:
            self.get_logger().info("✓ Starting inference")

        torch.set_grad_enabled(False)

        try:
            while rclpy.ok():
                start = time.time()

                rclpy.spin_once(self, timeout_sec=0.0)
                obs_np = self._build_observation()

                action_tensor = predict_action(
                    observation=obs_np,
                    policy=self.policy,
                    device=self.device,
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                    use_amp=self.policy.config.use_amp,
                    task=self.args.task,
                    robot_type="so101_follower",
                )

                action = action_tensor.cpu().numpy().astype(np.float32)
                action_normalized = action

                # Scale & convert
                action_scaled = action_normalized * self.action_scale
                action_radians = convert_so101_to_radians(action_scaled)

                # Temporal smoothing (EMA)
                if self.smoothed_action is None:
                    self.smoothed_action = action_radians.copy()
                else:
                    self.smoothed_action = (
                        self.smoothing_alpha * action_radians
                        + (1 - self.smoothing_alpha) * self.smoothed_action
                    )

                # Rerun 시각화 로깅
                self._log_visualization(
                    observation=obs_np,
                    action_normalized=action_normalized,
                    action_scaled=action_scaled,
                    action_radians=action_radians,
                )

                self._publish_joint_state(self.smoothed_action)
                self.step_counter += 1

                if (
                    self.max_steps is not None
                    and self.step_counter >= self.max_steps
                ):
                    self.get_logger().info("Reached max steps")
                    break

                # Timing
                elapsed = time.time() - start
                sleep = self.dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            self.get_logger().info("Interrupted")

        finally:
            if self.cap_top:
                self.cap_top.release()
            if self.cap_wrist:
                self.cap_wrist.release()
            self.get_logger().info("Shutdown complete")

# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--pretrained_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument("--top_camera_index", type=int, default=0)
    p.add_argument("--wrist_camera_index", type=int, default=1)
    p.add_argument("--camera_width", type=int, default=640)
    p.add_argument("--camera_height", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--robot_id", type=str, default="dsr01")
    p.add_argument(
        "--use_amp",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=None,
    )
    p.add_argument("--smoothing_alpha", type=float, default=0.3)
    p.add_argument("--action_scale", type=float, default=1.0)
    p.add_argument("--display_data", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.pretrained_path)

    if not path.is_dir():
        raise FileNotFoundError(
            f"pretrained_path must be a directory: {path}"
        )

    os.environ.setdefault("ROS_DOMAIN_ID", "0")

    rclpy.init()
    node: Optional[ACTDoosanBridge] = None

    try:
        node = ACTDoosanBridge(args)
        node.spin()

    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

