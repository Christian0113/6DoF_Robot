import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import rdplib.inverse_kinematics as ik
import rdplib.plot_helpers as ph
import rdplib.robot as rb
import rdplib.waypoint_generator as wp

assert sys.version_info >= (3, 10)

WAYPOINT_FILE = "waypoints_da.yaml"
DRAW_SEGMENT_DURATION = 0.12
PEN_UP_SEGMENT_DURATION = 0.25
APPROACH_DURATION = 1.0
DT = 0.001


def interpolate_joint_segment(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cubic polynomial joint interpolation (zero velocity at endpoints)."""
    num_steps = max(int(duration / dt), 2)
    delta_t = (num_steps - 1) * dt

    a = 2.0 / delta_t ** 3 * (q_start - q_end)
    b = 3.0 / delta_t ** 2 * (q_end - q_start)

    q_traj = np.zeros((q_start.size, num_steps))
    dot_q_traj = np.zeros_like(q_traj)
    ddot_q_traj = np.zeros_like(q_traj)
    time = dt * np.arange(num_steps)

    for k in range(num_steps):
        t = k * dt
        q_traj[:, k] = a * t ** 3 + b * t ** 2 + q_start
        dot_q_traj[:, k] = 3.0 * a * t ** 2 + 2.0 * b * t
        ddot_q_traj[:, k] = 6.0 * a * t + 2.0 * b

    return time, q_traj, dot_q_traj, ddot_q_traj


def build_viewer_data(
    robot: rb.Robot,
    time: np.ndarray,
    q_traj: np.ndarray,
    dot_q_traj: np.ndarray,
    ddot_q_traj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run forward kinematics and assemble viewer / analysis buffers."""
    num_steps = q_traj.shape[1]
    r_tcp_traj = np.zeros((3, num_steps))
    v_tcp_traj = np.zeros((3, num_steps))
    a_tcp_traj = np.zeros((3, num_steps))
    data = np.zeros((num_steps, 12 * robot.size + 1))

    q = np.zeros(robot.size)
    dot_q = np.zeros(robot.size)
    ddot_q = np.zeros(robot.size)

    for k in range(num_steps):
        q[:] = q_traj[:, k]
        dot_q[:] = dot_q_traj[:, k]
        ddot_q[:] = ddot_q_traj[:, k]

        robot.calculate_positions(q)
        robot.calculate_velocities(dot_q)
        robot.calculate_accelerations(ddot_q)
        robot.calculate_jacobis()

        r_tcp_traj[:, k] = robot.r_TCP__0
        v_tcp_traj[:, k] = robot.v_TCP__0
        a_tcp_traj[:, k] = robot.a_TCP__0

        data[k, 0] = time[k]
        for link_idx, link in enumerate(robot.links):
            base_col = link_idx * 12 + 1
            data[k, base_col:base_col + 3] = link.r_i__0
            data[k, base_col + 3:base_col + 12] = np.reshape(
                np.transpose(link.A_i0), (1, 9)
            )

    return data, r_tcp_traj, v_tcp_traj, a_tcp_traj


def _segment_duration(pen: str) -> float:
    if pen == "up":
        return PEN_UP_SEGMENT_DURATION
    return DRAW_SEGMENT_DURATION


def solve_writing_trajectory(
    robot: rb.Robot,
    waypoint_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Solve IK along stroke waypoints and build a joint-space trajectory."""
    if not os.path.isfile(waypoint_path):
        wp.save_da_waypoints(waypoint_path)

    segments = wp.load_stroke_segments(waypoint_path)
    targets: list[np.ndarray] = []
    labels: list[dict] = []

    for segment in segments:
        for col in range(segment["points"].shape[1]):
            targets.append(segment["points"][:, col])
            labels.append({"name": segment["name"], "pen": segment["pen"]})

    tcp_waypoints__0 = np.column_stack(targets)
    q_waypoints, success_flags = ik.solve_ik_waypoints(
        robot,
        tcp_waypoints__0,
        q_init=robot.q_home,
        q_reference=robot.q_comfort,
    )

    print("IK results for writing 大:")
    for idx, (label, success) in enumerate(zip(labels, success_flags)):
        robot.calculate_positions(q_waypoints[:, idx])
        err = np.linalg.norm(targets[idx] - robot.r_TCP__0)
        print(
            f"  {idx + 1:2d} [{label['pen']:4s}] {label['name']:14s} "
            f"success={success}, err={err * 1000:.2f} mm"
        )

    time_segments: list[np.ndarray] = []
    q_segments: list[np.ndarray] = []
    dot_q_segments: list[np.ndarray] = []
    ddot_q_segments: list[np.ndarray] = []

    time_seg, q_seg, dot_q_seg, ddot_q_seg = interpolate_joint_segment(
        robot.q_home,
        q_waypoints[:, 0],
        APPROACH_DURATION,
        DT,
    )
    time_segments.append(time_seg)
    q_segments.append(q_seg)
    dot_q_segments.append(dot_q_seg)
    ddot_q_segments.append(ddot_q_seg)

    for idx in range(q_waypoints.shape[1] - 1):
        duration = _segment_duration(labels[idx + 1]["pen"])
        time_seg, q_seg, dot_q_seg, ddot_q_seg = interpolate_joint_segment(
            q_waypoints[:, idx],
            q_waypoints[:, idx + 1],
            duration,
            DT,
        )
        time_seg = time_seg + time_segments[-1][-1] + DT
        time_segments.append(time_seg)
        q_segments.append(q_seg)
        dot_q_segments.append(dot_q_seg)
        ddot_q_segments.append(ddot_q_seg)

    time = np.concatenate(time_segments)
    q_traj = np.concatenate(q_segments, axis=1)
    dot_q_traj = np.concatenate(dot_q_segments, axis=1)
    ddot_q_traj = np.concatenate(ddot_q_segments, axis=1)
    return time, q_traj, dot_q_traj, ddot_q_traj, labels


def main() -> None:
    robot = rb.robot_factory(os.path.join("robot_config_files", "ABB_GoFa.yaml"))
    waypoint_path = os.path.join(WAYPOINT_FILE)

    time, q_traj, dot_q_traj, ddot_q_traj, _ = solve_writing_trajectory(
        robot,
        waypoint_path,
    )

    data, r_tcp_traj, v_tcp_traj, a_tcp_traj = build_viewer_data(
        robot, time, q_traj, dot_q_traj, ddot_q_traj
    )

    np.savetxt("trajectory.csv", data, delimiter="\t", fmt="%.4f")
    print(f"Saved trajectory.csv with {data.shape[0]} samples ({time[-1]:.2f} s).")

    ph.plot_path_3d(r_tcp_traj, "TCP Path: writing 大")
    ph.plot_angles(np.degrees(q_traj), time, "q", "Joint Angles (writing 大)", "deg")
    ph.plot_cartesian_trajectory(v_tcp_traj, time, "v_TCP", "Velocity of the TCP", "m/s")
    ph.plot_cartesian_trajectory(a_tcp_traj, time, "a_TCP", "Acceleration of the TCP", "m/s²")
    plt.show()


if __name__ == "__main__":
    main()
