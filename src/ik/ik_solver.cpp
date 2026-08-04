#include "ik/ik_solver.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace ik {

namespace {
pinocchio::Model buildModelFromUrdf(const std::string& urdf_path) {
  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdf_path, model);
  return model;
}
}  // namespace

RobotModel::RobotModel(const std::string& urdf_path, const std::string& ee_frame)
    : urdf_path_(urdf_path),
      model_(buildModelFromUrdf(urdf_path)),
      data_(model_),
      ee_id_(model_.getFrameId(ee_frame)) {}

namespace detail {

// Bound-constrained linear least squares via active-set: 매 반복마다 (a) 현재 활성 집합을 고정한
// 채 자유 변수만 정규방정식(QR)으로 풀고, (b) 그 결과가 경계를 벗어나면 가장 심하게 벗어난 변수
// 하나를 활성 집합에 추가, (c) 벗어난 게 없으면 KKT 조건(활성 변수의 라그랑주 승수 부호)을 확인해서
// 잘못 고정된 변수를 하나 풀어줌. (b)/(c) 둘 다 없으면 수렴. scipy.optimize.lsq_linear의
// 'bvls' 방식과 동일한 알고리즘.
Eigen::VectorXd solveBoundedLsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                 const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                 int max_iterations) {
  const int n = static_cast<int>(A.cols());
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  std::vector<bool> active(n, false);
  Eigen::VectorXd active_value = Eigen::VectorXd::Zero(n);

  for (int iter = 0; iter < max_iterations; ++iter) {
    std::vector<int> free_idx;
    free_idx.reserve(n);
    for (int i = 0; i < n; ++i) {
      if (!active[i]) free_idx.push_back(i);
    }

    Eigen::VectorXd b_eff = b;
    for (int i = 0; i < n; ++i) {
      if (active[i]) b_eff -= A.col(i) * active_value[i];
    }

    Eigen::VectorXd x_free;
    if (!free_idx.empty()) {
      Eigen::MatrixXd A_free(A.rows(), static_cast<int>(free_idx.size()));
      for (size_t k = 0; k < free_idx.size(); ++k) {
        A_free.col(static_cast<int>(k)) = A.col(free_idx[k]);
      }
      x_free = A_free.colPivHouseholderQr().solve(b_eff);
    }

    for (size_t k = 0; k < free_idx.size(); ++k) {
      x[free_idx[k]] = x_free[static_cast<int>(k)];
    }
    for (int i = 0; i < n; ++i) {
      if (active[i]) x[i] = active_value[i];
    }

    int worst_idx = -1;
    double worst_violation = 0.0;
    double worst_bound_value = 0.0;
    for (int idx : free_idx) {
      double violation = 0.0;
      double bound_value = 0.0;
      if (x[idx] < lb[idx]) {
        violation = lb[idx] - x[idx];
        bound_value = lb[idx];
      } else if (x[idx] > ub[idx]) {
        violation = x[idx] - ub[idx];
        bound_value = ub[idx];
      }
      if (violation > worst_violation) {
        worst_violation = violation;
        worst_idx = idx;
        worst_bound_value = bound_value;
      }
    }
    if (worst_idx >= 0) {
      active[worst_idx] = true;
      active_value[worst_idx] = worst_bound_value;
      x[worst_idx] = worst_bound_value;
      continue;
    }

    Eigen::VectorXd residual = A * x - b;
    Eigen::VectorXd gradient = A.transpose() * residual;
    int release_idx = -1;
    for (int i = 0; i < n; ++i) {
      if (!active[i]) continue;
      const bool at_lower = active_value[i] <= lb[i] + 1e-12;
      if (at_lower && gradient[i] < -1e-9) {
        release_idx = i;
        break;
      }
      if (!at_lower && gradient[i] > 1e-9) {
        release_idx = i;
        break;
      }
    }
    if (release_idx >= 0) {
      active[release_idx] = false;
      continue;
    }

    return x;
  }

  // 최대 반복 안에 못 끝났으면(이론상 n=8 문제에서는 발생하지 않아야 함) 최소한 제약은
  // 어기지 않도록 clamp해서 반환.
  return x.cwiseMax(lb).cwiseMin(ub);
}

}  // namespace detail

Eigen::VectorXd jointLimitAvoidanceDq(const RobotModel& rm, const Eigen::VectorXd& q, double k_pull) {
  const Eigen::VectorXd& lower = rm.model().lowerPositionLimit;
  const Eigen::VectorXd& upper = rm.model().upperPositionLimit;
  const Eigen::VectorXd q_mid = (lower + upper) / 2.0;
  const Eigen::VectorXd q_range = upper - lower;
  const Eigen::VectorXd gradient = 2.0 * (q - q_mid).array() / q_range.array().square();
  return -k_pull * gradient;
}

Eigen::Vector3d eePosition(RobotModel& rm, const Eigen::VectorXd& q) {
  pinocchio::forwardKinematics(rm.model(), rm.data(), q);
  pinocchio::updateFramePlacements(rm.model(), rm.data());
  return rm.data().oMf[rm.eeId()].translation();
}

double eeDownAlignment(RobotModel& rm, const Eigen::VectorXd& q) {
  pinocchio::forwardKinematics(rm.model(), rm.data(), q);
  pinocchio::updateFramePlacements(rm.model(), rm.data());
  // 회전행렬의 3번째 열이 로컬 Z축을 world 좌표로 표현한 벡터(R * [0,0,1]^T = R.col(2)).
  const Eigen::Vector3d local_z_in_world = rm.data().oMf[rm.eeId()].rotation().col(2);
  return local_z_in_world.dot(Eigen::Vector3d(0.0, 0.0, -1.0));
}

Eigen::VectorXd eeOrientationAlignmentDq(RobotModel& rm, const Eigen::VectorXd& q, double k_pull,
                                          double eps) {
  const double s0 = eeDownAlignment(rm, q);
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(rm.model().nq);

  std::vector<int> indices = {kTorsoJointIndex};
  for (int idx : kLeftArmIndices) indices.push_back(idx);

  for (int idx : indices) {
    Eigen::VectorXd q_pert = q;
    q_pert[idx] += eps;
    const double s_pert = eeDownAlignment(rm, q_pert);
    const double grad = (s_pert - s0) / eps;
    // (s - (-1))^2 = (s+1)^2를 최소화하는 gradient descent 방향
    // (elbowTorsoAvoidanceDq의 (threshold-d0)^2 최소화와 동일한 형태).
    dq[idx] = -k_pull * 2.0 * (s0 + 1.0) * grad;
  }
  return dq;
}

IkStepResult ikStep(RobotModel& rm, const Eigen::Vector3d& target_pos, const Eigen::VectorXd& q,
                     double damping, const IkStepParams& params,
                     const CollisionCheckFn& collision_checker) {
  const pinocchio::Model& model = rm.model();
  pinocchio::Data& data = rm.data();
  const int nq = model.nq;

  const Eigen::Vector3d current_pos = eePosition(rm, q);
  const Eigen::Vector3d error = target_pos - current_pos;
  const double error_norm = error.norm();

  IkStepResult result;
  if (error_norm < params.tol) {
    result.q = q;
    result.converged = true;
    result.damping = damping;
    result.hard_rejected = false;
    return result;
  }

  Eigen::Matrix<double, 6, Eigen::Dynamic> J6 =
      Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, model.nv);
  pinocchio::computeFrameJacobian(model, data, q, rm.eeId(), pinocchio::LOCAL_WORLD_ALIGNED, J6);
  const Eigen::MatrixXd jacobian_pos = J6.topRows(3);

  const Eigen::VectorXd joint_weights =
      params.joint_weights.size() == nq ? params.joint_weights : Eigen::VectorXd::Ones(nq);
  const Eigen::VectorXd secondary_dq =
      params.secondary_dq.size() == nq ? params.secondary_dq : Eigen::VectorXd::Zero(nq);

  // 1차 임무(위치)만으로 bounded LSQ를 풀어서 dq_primary를 구함. damping/joint_weight_scale
  // 정규화는 "1차 임무를 어떻게 푸는가"에 대한 것이지 2차목표가 아니라서 여기 포함.
  Eigen::MatrixXd A_primary = Eigen::MatrixXd::Zero(3 + nq + nq, nq);
  A_primary.topRows(3) = jacobian_pos;
  A_primary.middleRows(3, nq) = damping * Eigen::MatrixXd::Identity(nq, nq);
  Eigen::MatrixXd weight_block = Eigen::MatrixXd::Zero(nq, nq);
  weight_block.diagonal() = params.joint_weight_scale * joint_weights.array().sqrt().matrix();
  A_primary.bottomRows(nq) = weight_block;

  Eigen::VectorXd b_primary = Eigen::VectorXd::Zero(3 + nq + nq);
  b_primary.topRows(3) = error;

  Eigen::VectorXd lb = model.lowerPositionLimit - q;
  Eigen::VectorXd ub = model.upperPositionLimit - q;
  for (int idx : kRightArmIndices) {
    lb[idx] = -1e-9;
    ub[idx] = 1e-9;
  }

  const Eigen::VectorXd dq_primary = detail::solveBoundedLsq(A_primary, b_primary, lb, ub);

  // 2차목표(secondary_dq, 예: 그리퍼 방향 정렬)는 절대 1차 임무(위치)를 건드리지
  // 않도록 위치 자코비안의 null-space로 투영해서만 더함(Nakamura의 damped
  // null-space projection과 동일한 아이디어: N = I - J^+J, J^+ = J^T(JJ^T + eps*I)^-1).
  // 이전엔 secondary_dq를 같은 최소자승 문제에 그냥 얹어서 위치 수렴을 방해했음
  // (자유도가 4개뿐이라 방향 보정 방향 = 위치를 흔드는 방향이기도 했음) - 이제는
  // J_pos @ (N @ anything) ≈ 0이라 위치 결과가 secondary_dq 유무와 무관하게 유지됨.
  Eigen::VectorXd dq_full = dq_primary;
  if (secondary_dq.squaredNorm() > 0.0 && params.secondary_gain > 0.0) {
    constexpr double kNullSpaceRegularization = 1e-8;
    const Eigen::Matrix3d jjt_damped = jacobian_pos * jacobian_pos.transpose() +
                                        kNullSpaceRegularization * Eigen::Matrix3d::Identity();
    const Eigen::MatrixXd jacobian_pos_pinv = jacobian_pos.transpose() * jjt_damped.inverse();
    const Eigen::MatrixXd null_space_projector =
        Eigen::MatrixXd::Identity(nq, nq) - jacobian_pos_pinv * jacobian_pos;
    dq_full += null_space_projector * (params.secondary_gain * secondary_dq);
    // dq_primary는 이미 bound를 만족하는 상태였는데 투영된 2차 보정을 더하면 다시
    // 넘을 수 있음 - 엄밀한 재-QP 대신 표준 null-space IK 구현들처럼 clamp만 함.
    dq_full = dq_full.cwiseMax(lb).cwiseMin(ub);
  }

  const Eigen::VectorXd dq = params.alpha * dq_full;

  Eigen::VectorXd q_candidate(model.nq);
  pinocchio::integrate(model, q, dq, q_candidate);

  if (collision_checker && collision_checker(q_candidate)) {
    result.q = q;
    result.converged = false;
    result.damping = std::min(params.max_damping, damping * params.damping_increase_factor);
    result.hard_rejected = true;
    return result;
  }

  pinocchio::forwardKinematics(model, data, q_candidate);
  pinocchio::updateFramePlacements(model, data);
  const double candidate_error_norm = (target_pos - data.oMf[rm.eeId()].translation()).norm();

  const double predicted_reduction =
      error_norm * error_norm - (jacobian_pos * dq - error).squaredNorm();
  const double actual_reduction = error_norm * error_norm - candidate_error_norm * candidate_error_norm;
  const double rho = predicted_reduction < 1e-12 ? -1.0 : actual_reduction / predicted_reduction;

  if (rho > params.trust_region_good_ratio) {
    result.q = q_candidate;
    result.converged = false;
    result.damping = std::max(params.min_damping, damping * params.damping_decrease_factor);
    result.hard_rejected = false;
    return result;
  }

  if (rho > params.trust_region_acceptable_ratio) {
    result.q = q_candidate;
    result.converged = false;
    result.damping = damping;
    result.hard_rejected = false;
    return result;
  }

  result.q = q;
  result.converged = false;
  result.damping = std::min(params.max_damping, damping * params.damping_increase_factor);
  result.hard_rejected = false;
  return result;
}

}  // namespace ik
