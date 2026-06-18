"""
Inverse kinematics solvers for Unitree G1 robot.
"""
import time
import numpy as np
import casadi as cas
import pinocchio as pin
from pinocchio import casadi as cpin
from typing import Tuple, Optional

from .urdf_model import G1URDFModel
from .joint_config import get_q_indices, G1JointGroup, G1JointIndex, G1JointConfiguration


class DualArmIK:
    """
    Dual-arm inverse kinematics solver using CasADi optimization.

    Solves for both arm configurations simultaneously given target poses
    for left and right end-effectors.
    """

    def __init__(self):
        self.urdf_model = G1URDFModel(reduced=True)
        self.model = self.urdf_model.reduced_robot.model
        self.data = self.urdf_model.reduced_robot.data

        # Derive q-index sets from model + joint_config
        self.waist_q_idx      = get_q_indices([G1JointIndex.WaistYaw], self.model)
        self.left_q_idx       = get_q_indices(G1JointGroup.LEFT_ARM,   self.model)
        self.right_q_idx      = get_q_indices(G1JointGroup.RIGHT_ARM,  self.model)
        self.both_arms_q_idx  = get_q_indices(G1JointGroup.BOTH_ARMS,  self.model)
        self.upper_body_q_idx = get_q_indices(G1JointGroup.UPPER_BODY, self.model)

        # Build operative joint bounds (joint_config overrides URDF for upper-body joints)
        self.q_lb_default, self.q_ub_default = self._build_q_bounds()

        # Setup CasADi symbolic IK
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Symbolic variables
        self.cq = cas.SX.sym("q", self.model.nq, 1)
        self.cTf_left = cas.SX.sym("tf_left", 4, 4)
        self.cTf_right = cas.SX.sym("tf_right", 4, 4)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get end-effector frame IDs
        self.left_ee_id = self.model.getFrameId("left_ee")
        self.right_ee_id = self.model.getFrameId("right_ee")

        # Define per-arm error functions
        self.left_trans_error = cas.Function(
            "left_trans_error",
            [self.cq, self.cTf_left],
            [self.cdata.oMf[self.left_ee_id].translation - self.cTf_left[:3, 3]],
        )
        self.right_trans_error = cas.Function(
            "right_trans_error",
            [self.cq, self.cTf_right],
            [self.cdata.oMf[self.right_ee_id].translation - self.cTf_right[:3, 3]],
        )
        # Use Frobenius norm -- otherwise gradient issues?
        self.left_rot_error = cas.Function(
            "left_rot_error",
            [self.cq, self.cTf_left],
            [cas.reshape(self.cdata.oMf[self.left_ee_id].rotation - self.cTf_left[:3, :3], (9, 1))],
        )
        self.right_rot_error = cas.Function(
            "right_rot_error",
            [self.cq, self.cTf_right],
            [cas.reshape(self.cdata.oMf[self.right_ee_id].rotation - self.cTf_right[:3, :3], (9, 1))],
        )

        # Setup optimization
        self.opti = cas.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)
        self.param_tf_left = self.opti.parameter(4, 4)
        self.param_tf_right = self.opti.parameter(4, 4)

        # Per-DOF bound parameters (set to model limits by default, locked joints get lb==ub)
        self.param_q_lb = self.opti.parameter(self.model.nq)
        self.param_q_ub = self.opti.parameter(self.model.nq)
        self.opti.set_value(self.param_q_lb, self.q_lb_default)
        self.opti.set_value(self.param_q_ub, self.q_ub_default)

        # Per-arm EE cost weight parameters
        self.param_left_trans_w  = self.opti.parameter()
        self.param_right_trans_w = self.opti.parameter()
        self.param_left_rot_w    = self.opti.parameter()
        self.param_right_rot_w   = self.opti.parameter()
        for p in [self.param_left_trans_w, self.param_right_trans_w,
                  self.param_left_rot_w,   self.param_right_rot_w]:
            self.opti.set_value(p, 1.0)

        # Cost terms (per-arm, weighted)
        left_trans_err  = self.left_trans_error(self.var_q,  self.param_tf_left)
        right_trans_err = self.right_trans_error(self.var_q, self.param_tf_right)
        left_rot_err    = self.left_rot_error(self.var_q,    self.param_tf_left)
        right_rot_err   = self.right_rot_error(self.var_q,   self.param_tf_right)

        trans_cost  = (self.param_left_trans_w  * cas.sumsqr(left_trans_err) +
                       self.param_right_trans_w * cas.sumsqr(right_trans_err))
        rot_cost    = (self.param_left_rot_w    * cas.sumsqr(left_rot_err) +
                       self.param_right_rot_w   * cas.sumsqr(right_rot_err))
        q_neutral   = cas.DM(pin.neutral(self.model))
        reg_cost    = cas.sumsqr(self.var_q - q_neutral)
        smooth_cost = cas.sumsqr(self.var_q - self.var_q_last)

        # Extra waist penalty discourages motion the arms can do
        waist_idx = self.waist_q_idx[0]
        waist_reg_cost    = (self.var_q[waist_idx] - q_neutral[waist_idx]) ** 2
        waist_smooth_cost = (self.var_q[waist_idx] - self.var_q_last[waist_idx]) ** 2
        self.param_reg_waist_w = self.opti.parameter()
        self.param_smooth_waist_w = self.opti.parameter()
        self.opti.set_value(self.param_reg_waist_w, 0.2)
        self.opti.set_value(self.param_smooth_waist_w, 5.0)

        # Per-DOF bounds constraint
        self.opti.subject_to(self.opti.bounded(self.param_q_lb, self.var_q, self.param_q_ub))

        # Objective function (weighted sum)
        self.opti.minimize(
            50 * trans_cost +
            1.0 * rot_cost +
            0.02 * reg_cost +
            0.5 * smooth_cost +
            self.param_reg_waist_w * waist_reg_cost +
            self.param_smooth_waist_w * waist_smooth_cost
        )

        # Solver options
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 20,
                'tol': 1e-4
            },
            'print_time': False,
            'calc_lam_p': False
        }
        self.opti.solver("ipopt", opts)

        # Pre-warm: trigger IPOPT's first-call JIT/license overhead now, not at runtime
        print("IK solver: pre-warming IPOPT...")
        q_neutral = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_neutral)
        left_neutral  = self.data.oMf[self.left_ee_id].homogeneous.copy()
        right_neutral = self.data.oMf[self.right_ee_id].homogeneous.copy()
        self.solve(left_neutral, right_neutral, q_current=q_neutral)
        print("IK solver: ready.")

    def _build_q_bounds(self):
        """Build operative joint bounds: URDF as base, overridden by joint_config for upper-body."""
        lb = self.model.lowerPositionLimit.copy()
        ub = self.model.upperPositionLimit.copy()
        limits = G1JointConfiguration.get_limits_arrays(G1JointGroup.UPPER_BODY)
        for i, q_idx in enumerate(self.upper_body_q_idx):
            lb[q_idx] = limits['q_min'][i]
            ub[q_idx] = limits['q_max'][i]
        return lb, ub

    def solve(
        self,
        left_ee_pose: np.ndarray,
        right_ee_pose: np.ndarray,
        q_current: Optional[np.ndarray] = None,
        dq_current: Optional[np.ndarray] = None,
        dt: float = 0.02,
        free_joints: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve dual-arm IK.

        Args:
            left_ee_pose:  Left end-effector target pose (4x4)
            right_ee_pose: Right end-effector target pose (4x4)
            q_current:     Current joint positions for warm start (nq,)
            dq_current:    Current joint velocities (nv,)
            dt:            Timestep for computing finite difference (float)
            free_joints:   List of q-vector indices to optimize; None → both arms only.
                           Locked joints are pinned to q_current via per-DOF bounds.

        Returns:
            (q_sol, dq_sol, tau_ff)
        """
        if free_joints is None:
            free_joints = self.both_arms_q_idx
        free_set = set(free_joints)

        q_init  = q_current  if q_current  is not None else pin.neutral(self.model)
        dq_init = dq_current if dq_current is not None else np.zeros(self.model.nv)

        # Per-DOF bounds: locked joints get lb == ub == q_current
        lb = self.q_lb_default.copy()
        ub = self.q_ub_default.copy()
        for i in range(self.model.nq):
            if i not in free_set:
                lb[i] = ub[i] = float(q_init[i])
        self.opti.set_value(self.param_q_lb, lb)
        self.opti.set_value(self.param_q_ub, ub)

        # Per-arm EE cost weights: zero out arms whose joints are all locked
        left_w  = 1.0 if any(i in free_set for i in self.left_q_idx)  else 0.0
        right_w = 1.0 if any(i in free_set for i in self.right_q_idx) else 0.0
        self.opti.set_value(self.param_left_trans_w,  left_w)
        self.opti.set_value(self.param_right_trans_w, right_w)
        self.opti.set_value(self.param_left_rot_w,    left_w)
        self.opti.set_value(self.param_right_rot_w,   right_w)

        self.opti.set_initial(self.var_q, q_init)
        self.opti.set_value(self.param_tf_left,  left_ee_pose)
        self.opti.set_value(self.param_tf_right, right_ee_pose)
        self.opti.set_value(self.var_q_last, q_init)

        try:
            self.opti.solve()
            q_sol = self.opti.value(self.var_q)
        except Exception as e:
            print(f"IK solver failed to converge: {e}")
            try:
                q_sol = self.opti.debug.value(self.var_q)
            except Exception:
                q_sol = q_init.copy()

        # Clamp locked joints to exact q_init (numerical safety)
        locked = [i for i in range(self.model.nq) if i not in free_set]
        q_sol[locked] = q_init[locked]

        # Compute dq as finite difference
        dq_sol = (q_sol - q_current) / dt if q_current is not None else dq_init

        # Update forward kinematics at q_sol
        pin.framesForwardKinematics(self.model, self.data, q_sol)

        # Compute feedforward torques using RNEA
        tau_ff = pin.rnea(
            self.model, self.data,
            q_sol, dq_sol,
            np.zeros(self.model.nv),
        )

        return q_sol, dq_sol, tau_ff


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Dual-Arm IK Solver")
    print("=" * 70)

    print("\n[1] Initializing IK solver...")
    ik_solver = DualArmIK()
    print(f"    ✓ Reduced model loaded: {ik_solver.model.nq} DOF")
    print(f"    ✓ Left EE frame ID: {ik_solver.left_ee_id}")
    print(f"    ✓ Right EE frame ID: {ik_solver.right_ee_id}")

    # Test: Forward-Inverse consistency
    print("\n[2] Test: Forward-Inverse Consistency")
    print("    Generate random config → FK → IK → compare")

    n_tests = 10
    successes = 0
    position_errors = []
    elapsed_times = []

    for i in range(n_tests):
        start_time = time.time()

        q_test  = pin.randomConfiguration(ik_solver.model)
        dq_test = np.random.randn(ik_solver.model.nv) * 0.1

        pin.framesForwardKinematics(ik_solver.model, ik_solver.data, q_test)
        left_target  = ik_solver.data.oMf[ik_solver.left_ee_id].homogeneous
        right_target = ik_solver.data.oMf[ik_solver.right_ee_id].homogeneous

        q_sol, _, _ = ik_solver.solve(
            left_target, right_target,
            q_current=q_test,
            dq_current=dq_test,
        )

        pin.framesForwardKinematics(ik_solver.model, ik_solver.data, q_sol)
        left_achieved  = ik_solver.data.oMf[ik_solver.left_ee_id].homogeneous
        right_achieved = ik_solver.data.oMf[ik_solver.right_ee_id].homogeneous

        left_pos_error  = np.linalg.norm(left_target[:3, 3]  - left_achieved[:3, 3])
        right_pos_error = np.linalg.norm(right_target[:3, 3] - right_achieved[:3, 3])
        max_pos_error   = max(left_pos_error, right_pos_error)
        position_errors.append(max_pos_error)
        elapsed_times.append(time.time() - start_time)

        if max_pos_error < 0.01:
            successes += 1
        print(f"    {'✓' if max_pos_error < 0.01 else '✗'} Test {i+1}: pos_error={max_pos_error*1000:.2f}mm")

    print(f"\n    Success rate: {successes}/{n_tests} ({100*successes/n_tests:.1f}%)")
    print(f"    Avg position error: {np.mean(position_errors)*1000:.2f}mm")
    print(f"    Max position error: {np.max(position_errors)*1000:.2f}mm")
    print(f"    Avg solve time:     {np.mean(elapsed_times)*1000:.2f}ms")

    print("\n[3] Test: free_joints locking (right arm free, left arm locked)")
    q_neutral_main = pin.neutral(ik_solver.model)
    pin.framesForwardKinematics(ik_solver.model, ik_solver.data, q_neutral_main)
    left_neutral  = ik_solver.data.oMf[ik_solver.left_ee_id].homogeneous.copy()
    right_neutral = ik_solver.data.oMf[ik_solver.right_ee_id].homogeneous.copy()

    q_test = pin.randomConfiguration(ik_solver.model)
    q_sol, _, _ = ik_solver.solve(
        left_neutral, right_neutral,   # targets don't matter for locked left
        q_current=q_test,
        free_joints=ik_solver.right_q_idx,
    )
    left_locked  = np.allclose(q_sol[ik_solver.left_q_idx],  q_test[ik_solver.left_q_idx])
    waist_locked = np.allclose(q_sol[ik_solver.waist_q_idx], q_test[ik_solver.waist_q_idx])
    print(f"    {'✓' if left_locked  else '✗'} Left arm joints unchanged")
    print(f"    {'✓' if waist_locked else '✗'} Waist joint unchanged")

    print("\n[4] Test: waist free — moves when solving for a random target")
    # Run N trials from neutral; waist should move on at least most of them
    n_waist_trials = 10
    waist_moved = 0
    waist_deltas = []
    waist_tol = 1e-3  # [rad] — optimizer noise threshold

    for _ in range(n_waist_trials):
        q_rand = pin.randomConfiguration(ik_solver.model)
        pin.framesForwardKinematics(ik_solver.model, ik_solver.data, q_rand)
        left_rand  = ik_solver.data.oMf[ik_solver.left_ee_id].homogeneous.copy()
        right_rand = ik_solver.data.oMf[ik_solver.right_ee_id].homogeneous.copy()

        q_sol, _, _ = ik_solver.solve(
            left_rand, right_rand,
            q_current=q_neutral_main,
            free_joints=ik_solver.upper_body_q_idx,  # waist included
        )
        delta = abs(float(q_sol[ik_solver.waist_q_idx[0]]) - float(q_neutral_main[ik_solver.waist_q_idx[0]]))
        waist_deltas.append(delta)
        if delta > waist_tol:
            waist_moved += 1

    print(f"    Waist moved (>{waist_tol*1000:.0f}mrad) in {waist_moved}/{n_waist_trials} trials")
    print(f"    Waist delta range: {min(waist_deltas)*1000:.1f}–{max(waist_deltas)*1000:.1f} mrad")
    print(f"    {'✓' if waist_moved > 0 else '✗'} Waist is free to move when included in free_joints")

    print("\n" + "=" * 70)
    print("IK solver tests complete!")
    print("=" * 70)
