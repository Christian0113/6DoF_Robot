import numpy as np
from scipy.optimize import least_squares

from . import robot as rb


def compute_translational_jacobian(robot: rb.Robot) -> np.ndarray:
    """Geometric translational Jacobian of the TCP in frame 0.

    Column i corresponds to joint i+1 (self-z convention: rotation about z_i).
    """
    r_tcp = robot.r_TCP__0
    jacobian = np.zeros((3, robot.size))

    for link in robot.links:
        idx = link.dof - 1
        z_axis__0 = link.A_i0.T @ np.array([0.0, 0.0, 1.0])
        jacobian[:, idx] = np.cross(z_axis__0, r_tcp - link.r_i__0)

    return jacobian


def _wrap_to_limits(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    """Keep joint angles inside limits by adding multiples of 2*pi."""
    two_pi = 2.0 * np.pi
    wrapped = np.copy(q)

    for i in range(q.size):
        while wrapped[i] < q_min[i]:
            wrapped[i] += two_pi
        while wrapped[i] > q_max[i]:
            wrapped[i] -= two_pi

    return wrapped


def solve_ik_position(
    robot: rb.Robot,
    r_target__0: np.ndarray,
    q_init: np.ndarray | None = None,
    q_reference: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-5,
    damping: float = 0.05,
    step_gain: float = 0.8,
    null_gain: float = 0.15,
) -> tuple[np.ndarray, bool, float]:
    """Numerical position IK using damped least squares.

    Parameters
    ----------
    robot
        Robot model used for forward kinematics.
    r_target__0
        Desired TCP position in frame 0 [m].
    q_init
        Initial guess in joint space [rad]. Defaults to robot.q_home.
    q_reference
        Preferred posture for null-space regularization. Defaults to q_init.
    max_iter
        Maximum Newton iterations.
    tol
        Position error tolerance [m].
    damping
        Damping factor for the Jacobian pseudo-inverse.
    step_gain
        Step-size scaling for joint updates.
    null_gain
        Gain for null-space motion toward q_reference.

    Returns
    -------
    q, success, error
        Solved joint angles, convergence flag, final position error [m].
    """
    q = np.copy(q_init if q_init is not None else robot.q_home)
    q_ref = np.copy(q_reference if q_reference is not None else q)
    identity = np.eye(robot.size)

    for _ in range(max_iter):
        robot.calculate_positions(q)
        error = r_target__0 - robot.r_TCP__0
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return _wrap_to_limits(q, robot.q_min, robot.q_max), True, error_norm

        jacobian = compute_translational_jacobian(robot)
        jjt = jacobian @ jacobian.T + (damping ** 2) * np.eye(3)
        jacobian_pinv = jacobian.T @ np.linalg.inv(jjt)
        delta_q = step_gain * (jacobian_pinv @ error)

        null_projector = identity - jacobian_pinv @ jacobian
        delta_q += null_gain * (null_projector @ (q_ref - q))

        q = _wrap_to_limits(q + delta_q, robot.q_min, robot.q_max)

    robot.calculate_positions(q)
    final_error = np.linalg.norm(r_target__0 - robot.r_TCP__0)
    if final_error >= tol:
        q = _refine_ik_position(
            robot,
            r_target__0,
            q,
            tol,
        )
        robot.calculate_positions(q)
        final_error = np.linalg.norm(r_target__0 - robot.r_TCP__0)

    success = final_error < tol
    return _wrap_to_limits(q, robot.q_min, robot.q_max), success, final_error


def _refine_ik_position(
    robot: rb.Robot,
    r_target__0: np.ndarray,
    q_init: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Polish IK solution with Levenberg-Marquardt least squares."""

    def residual(q_vec: np.ndarray) -> np.ndarray:
        robot.calculate_positions(q_vec)
        return r_target__0 - robot.r_TCP__0

    result = least_squares(
        residual,
        q_init,
        bounds=(robot.q_min, robot.q_max),
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=1000,
    )
    return result.x


def solve_ik_waypoints(
    robot: rb.Robot,
    tcp_waypoints__0: np.ndarray,
    q_init: np.ndarray | None = None,
    q_reference: np.ndarray | None = None,
    **ik_kwargs,
) -> tuple[np.ndarray, list[bool]]:
    """Solve IK for a sequence of TCP positions using continuation."""
    num_waypoints = tcp_waypoints__0.shape[1]
    q_waypoints = np.zeros((robot.size, num_waypoints))
    success_flags: list[bool] = []

    q_seed = np.copy(q_init if q_init is not None else robot.q_home)
    q_ref = np.copy(q_reference if q_reference is not None else robot.q_comfort)

    for idx in range(num_waypoints):
        q_seed, success, _ = solve_ik_position(
            robot,
            tcp_waypoints__0[:, idx],
            q_init=q_seed,
            q_reference=q_ref,
            **ik_kwargs,
        )
        q_waypoints[:, idx] = q_seed
        success_flags.append(success)

    return q_waypoints, success_flags
