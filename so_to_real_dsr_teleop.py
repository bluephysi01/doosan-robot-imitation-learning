#!/usr/bin/env python3
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import JointState

from dsr_msgs2.srv import MoveJoint
from dsr_example.simple.gripper_drl_controller import GripperController


class SoToRealDsrTeleop(Node):
    def __init__(self):
        super().__init__("so_to_real_dsr_teleop")

        self.robot_id = self.declare_parameter("robot_id", "dsr01").value
        self.input_joint_topic = self.declare_parameter("input_joint_topic", "/joint_states").value
        self.sim_joint_topic = f"/{self.robot_id}/joint_states"

        # Arm control rate
        self.command_period = 0.15
        self.stale_timeout = 0.75
        self.joint_tolerance_deg = 0.3

        # Gripper update ONLY every 1 sec
        self.gripper_command_period = 1.0
        # Gripper tolerances before considering update
        self.gripper_tolerance = 50.0  # bigger tolerance since 100-step quantization

        self._state_lock = threading.Lock()
        self._latest_joints_deg = None
        self._latest_gripper_stroke = None
        self._latest_stamp = 0.0

        self._last_sent_joints = None
        self._last_move_time = 0.0

        self._last_gripper_stroke = None
        self._last_gripper_command_time = 0.0
        self._current_gripper_stroke = 0

        # Timer controlling ARM + GRIPPER update
        self._command_timer = self.create_timer(self.command_period, self._command_timer_cb)

        # Doosan moveJ service
        self.client = self.create_client(MoveJoint, f"/{self.robot_id}/motion/move_joint")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for move_joint service...")

        # DRL gripper controller
        self.gripper = GripperController(self, self.robot_id, 0)

        # Publish to IsaacSim arm + gripper joint state
        self.sim_pub = self.create_publisher(JointState, self.sim_joint_topic, 10)
        self.isaac_gripper_joint = "rh_r1"

        # Subscribe SO arm
        self.subscription = self.create_subscription(
            JointState, self.input_joint_topic, self.cb, 20
        )

        self.dsr_joint_names = [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6"
        ]

        self.get_logger().info("SO → REAL Doosan teleop (ARM + GRIPPER) STARTED.")


    def cb(self, msg):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        rot     = pos["Rotation"]
        pitch   = pos["Pitch"]
        elbow   = pos["Elbow"]
        w_pitch = pos["Wrist_Pitch"]
        w_roll  = pos["Wrist_Roll"]
        jaw     = pos["Jaw"]  # rad

        # SO → Doosan coordinate flip
        rot = -rot
        pitch = -pitch
        elbow = -elbow
        w_pitch = -w_pitch

        # Offsets
        rot += -math.pi/2
        elbow += -math.pi/2

        joints_rad = [rot, pitch, elbow, 0.0, w_pitch, w_roll]
        joints_deg = [math.degrees(j) for j in joints_rad]

        stroke = self._jaw_to_stroke(jaw)

        now = time.time()
        with self._state_lock:
            self._latest_joints_deg = joints_deg
            self._latest_gripper_stroke = stroke
            self._latest_stamp = now

        # publish to IsaacSim
        current_jaw = self._stroke_to_jaw(self._current_gripper_stroke)
        sim_msg = JointState()
        sim_msg.header.stamp = self.get_clock().now().to_msg()
        sim_msg.name = self.dsr_joint_names + [self.isaac_gripper_joint]
        sim_msg.position = joints_rad + [current_jaw]
        sim_msg.velocity = [0.0] * 7
        self.sim_pub.publish(sim_msg)


    def _command_timer_cb(self):
        now = time.time()
        with self._state_lock:
            joints = None if self._latest_joints_deg is None else list(self._latest_joints_deg)
            stroke = self._latest_gripper_stroke
            stamp = self._latest_stamp

        if (now - stamp) > self.stale_timeout:
            return

        # ARM update
        if joints:
            if (not self._last_sent_joints) or self._joints_changed(joints):
                if (now - self._last_move_time) >= self.command_period:
                    self._last_move_time = now
                    self._last_sent_joints = list(joints)
                    self.movej(joints)

        # GRIPPER update every 1 sec & quantized 100 steps
        if stroke is not None:
            if (now - self._last_gripper_command_time) >= self.gripper_command_period:
                if (self._last_gripper_stroke is None or
                    abs(stroke - self._last_gripper_stroke) >= self.gripper_tolerance):

                    self._maybe_move_gripper(stroke)
                    self._last_gripper_command_time = now


    def _joints_changed(self, new_joints):
        if not self._last_sent_joints:
            return True
        for prev, curr in zip(self._last_sent_joints, new_joints):
            if abs(curr - prev) > self.joint_tolerance_deg:
                return True
        return False


    # --------------------------------------------------------
    # ✨ Jaw→Stroke: 0~2.2 rad → 700~0 map + 100-step quantization
    # --------------------------------------------------------
    def _jaw_to_stroke(self, jaw_value: float) -> int:
        jaw_clamped = max(0.0, min(2.2, jaw_value))

        # linear mapping: jaw 0.0 → stroke 700, jaw 2.2 → stroke 0
        raw = 700.0 - (jaw_clamped / 2.2) * 700.0

        # 🔥 quantize stroke to 100-unit bins
        quantized = int(round(raw / 100.0) * 100)

        return max(0, min(700, quantized))


    # --------------------------------------------------------
    # ✨ Stroke→Jaw: inverse mapping of above
    # --------------------------------------------------------
    def _stroke_to_jaw(self, stroke_value: int) -> float:
        stroke_clamped = max(0, min(700, stroke_value))
        jaw_value = (700.0 - stroke_clamped) / 700.0 * 2.2
        return max(0.0, min(2.2, jaw_value))


    def _maybe_move_gripper(self, stroke):
        self.gripper.move(stroke)
        self._last_gripper_stroke = stroke
        self._current_gripper_stroke = stroke
        jaw_show = self._stroke_to_jaw(stroke)
        self.get_logger().info(f"✓ Gripper DRL sent: stroke={stroke}, jaw={jaw_show:.2f} rad")


    def movej(self, joint_values):
        req = MoveJoint.Request()
        req.pos = [float(v) for v in joint_values]
        req.vel = 15.0  # 속도 15% (원래 50)
        req.acc = 15.0  # 가속도 15% (원래 50)
        req.time = 0.0
        req.radius = 30.0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 0

        future = self.client.call_async(req)
        future.add_done_callback(self._on_movej_done)

    def _on_movej_done(self, future):
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().warn("MoveJoint request failed.")
        except Exception as exc:
            self.get_logger().error(f"MoveJoint exception: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = SoToRealDsrTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
