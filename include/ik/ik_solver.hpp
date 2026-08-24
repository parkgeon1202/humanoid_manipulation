#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

// scripts/lib/ik_solver.py의 C++ 포팅. bound-QP(joint limit + joint_weight_scale +
// null-space 2차목표를 하나로 합친 최소자승법) + trust-region accept/reject를 그대로 옮김.
// 이 모듈은 nq==nv(전부 1-DOF revolute 관절, quaternion/구형 관절 없음)를 전제함 —
// irc_man.urdf가 실제로 그러함(model.nq == model.nv == 8).
namespace ik {

// irc_man.urdf 조인트 순서: torso_yaw(0), left_shoulder_pitch(1), left_shoulder_roll(2),
// left_elbow_pitch(3), right_shoulder_pitch(4), right_shoulder_roll(5), right_elbow_pitch(6),
// tilt(7). scripts/lib/ik_solver.py의 동일 주석과 일치.
constexpr int kTorsoJointIndex = 0;
constexpr std::array<int, 3> kLeftArmIndices = {1, 2, 3};
constexpr std::array<int, 3> kRightArmIndices = {4, 5, 6};

constexpr double kDefaultDamping = 1e-3;
constexpr double kMinDamping = 1e-6;
constexpr double kMaxDamping = 1e2;
constexpr double kDampingDecreaseFactor = 0.5;
constexpr double kDampingIncreaseFactor = 2.0;
constexpr double kTrustRegionGoodRatio = 0.75;
constexpr double kTrustRegionAcceptableRatio = 0.1;

// pinocchio::Model/Data/ee frame id를 한 번만 로드해서 들고 있는 래퍼. pinocchio::Data는
// 재진입-safe하지 않으므로(Python의 `data = model.createData()`와 동일하게) 노드/테스트당
// 하나씩 생성해서 씀 - CollisionChecker도 자기 전용 Data를 별도로 들고 있음(공유 안 함).
class RobotModel {
 public:
  explicit RobotModel(const std::string& urdf_path, const std::string& ee_frame = "ee_link_1");

  const pinocchio::Model& model() const { return model_; }
  pinocchio::Data& data() { return data_; }
  pinocchio::FrameIndex eeId() const { return ee_id_; }
  // collision_model.hpp의 CollisionChecker가 링크 collision mesh를 다시 찾을 때(package://
  // 경로 해석용) 필요해서 보관해둠 - Python의 모듈 전역 urdf_path와 동일한 역할.
  const std::string& urdfPath() const { return urdf_path_; }

 private:
  std::string urdf_path_;
  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex ee_id_;
};

// ik_step에 넘기는 충돌 검사 콜백. collision_model.hpp의 CollisionChecker::isColliding을
// 감싸서 넘기면 됨(예: [&](const Eigen::VectorXd& q){ return checker.isColliding(q); }).
// nullptr(빈 std::function)이면 hard 충돌 체크를 아예 안 함.
using CollisionCheckFn = std::function<bool(const Eigen::VectorXd&)>;

struct IkStepParams {
  double tol = 1e-4;
  double alpha = 0.5;
  double min_damping = kMinDamping;
  double max_damping = kMaxDamping;
  double damping_decrease_factor = kDampingDecreaseFactor;
  double damping_increase_factor = kDampingIncreaseFactor;
  double trust_region_good_ratio = kTrustRegionGoodRatio;
  double trust_region_acceptable_ratio = kTrustRegionAcceptableRatio;
  // 비어있으면(size()==0) 전부 1.0으로 취급(Python의 joint_weights=None과 동일).
  Eigen::VectorXd joint_weights;
  double joint_weight_scale = 0.0;
  // 비어있으면(size()==0) 전부 0.0으로 취급(Python의 secondary_dq=None과 동일).
  Eigen::VectorXd secondary_dq;
  double secondary_gain = 0.0;
};

struct IkStepResult {
  Eigen::VectorXd q;
  bool converged = false;
  double damping = kDefaultDamping;
  // 이번 tick이 hard 충돌 반려(collision_checker가 true) 때문에 제자리였는지만 따로 알려줌 —
  // damping이 상한에 붙은 이유가 rho(선형근사 불신) 때문인지 충돌 때문인지 호출부가 구분하게
  // 해줌(예: torso-kick 탈출 로직이 진짜 충돌 정체만 감지하도록).
  bool hard_rejected = false;
};

// Liégeois(1977)식 관절 한계 회피 2차목표. 현재 어떤 호출부에서도 안 씀(Python도 주석 처리된
// 채로 남아있던 것과 동일하게, 포팅만 해두고 비활성 상태로 둠) — 관절 한계는 bound QP(lb/ub)가
// 이미 절대 못 넘게 막고 있어서 이건 순전히 "중앙으로 당기는 선호"일 뿐이라 지금은 불필요.
Eigen::VectorXd jointLimitAvoidanceDq(const RobotModel& rm, const Eigen::VectorXd& q,
                                       double k_pull = 1.0);

// q에 대한 forward kinematics를 갱신하고 ee 프레임의 월드 좌표를 반환. ikStep이 내부적으로
// 하던 것과 동일 - place_trajectory처럼 IK 없이 "지금 이 q에서 EE가 어디 있는지"만
// 필요한 외부 호출부가 재사용하도록 뺌.
Eigen::Vector3d eePosition(RobotModel& rm, const Eigen::VectorXd& q);

// ee_link_1의 로컬 Z축(그리퍼가 물건을 향해 뻗는 축)이 world +Z(수직 위)와 얼마나
// 정렬됐는지를 나타내는 스칼라. +1이면 완벽히 정렬(PICK의 top-down 그립에 맞는 방향 -
// ee_link_1 프레임 자체의 로컬 Z축 정의가 "물건 쪽"이 아니라 반대쪽을 향해있어서,
// world +Z 쪽으로 당겨야 실제로는 그리퍼가 아래를 보게 됨. 대화로 확인함), -1이면
// 정반대. 함수 이름은 "물리적으로 아래를 향하게 한다"는 의미로 그대로 둠 - dot 타겟은
// world +Z지만 결과적으로 하는 일은 여전히 top-down 정렬.
double eeDownAlignment(RobotModel& rm, const Eigen::VectorXd& q);

// torso_yaw + 왼팔 3관절에 대해, eeDownAlignment(q)를 +1(완전히 정렬)에 최대한 가깝게
// 만드는 방향으로 당기는 dq를 유한차분으로 근사해서 반환(secondary_dq로 바로 사용 가능).
// elbowTorsoAvoidanceDq(collision_model.hpp)와 동일한 유한차분 패턴이지만, threshold로
// 게이팅하지 않고 항상 당김(정렬은 "가까워지면"이 아니라 "항상 최대한" 원하는 목표라서).
// 오른팔/tilt는 이 정렬에 영향이 없어 계산하지 않음.
Eigen::VectorXd eeOrientationAlignmentDq(RobotModel& rm, const Eigen::VectorXd& q,
                                          double k_pull = 1.0, double eps = 1e-4);

// ee_link_1의 로컬 X축(그리퍼를 놓을 때 세우는 축)이 world +Z(수직 위)와 얼마나
// 정렬됐는지를 나타내는 스칼라. +1이면 완벽히 정렬(로컬 X가 그대로 위를 향함),
// -1이면 정반대. VIRTUAL_PLACE 그리퍼 오픈 시퀀스가 "놓는 자세"를 잡을 때 씀
// (fsm.cpp stepPlaceIk 참고) - eeDownAlignment(PICK의 top-down 그립용)와는 축/타겟
// 둘 다 달라서 별도 함수로 둠.
double eeXAxisUpAlignment(RobotModel& rm, const Eigen::VectorXd& q);

// torso_yaw + 왼팔 3관절에 대해, eeXAxisUpAlignment(q)를 +1(완전히 정렬)에 최대한
// 가깝게 만드는 방향으로 당기는 dq를 유한차분으로 근사해서 반환(eeOrientationAlignmentDq와
// 동일한 유한차분 패턴, 타겟만 +1). [현재 미사용 - jointLimitAvoidanceDq와 동일하게
// 포팅만 해두고 비활성 상태. stepPlaceIk는 매 tick 이 dq를 고정 크기로 그냥 더하는
// 대신 orientationAlignStep(damped least squares + trust-region)을 씀 - 고정 스텝
// 방식은 threshold를 못 넘는 자세에서 계속 진동하는 문제가 있었음(대화 참고)].
Eigen::VectorXd eeXAxisUpAlignmentDq(RobotModel& rm, const Eigen::VectorXd& q,
                                      double k_pull = 1.0, double eps = 1e-4);

// ee_link_1의 로컬 Y축이 world Y축과 얼마나 정렬됐는지를 나타내는 스칼라(+1이면 완벽히
// 정렬). eeXAxisUpAlignment(로컬 X축 -> world +Z, "놓는 자세"의 방향)는 orientation
// 3자유도 중 2개만 묶는데(축 하나를 특정 방향에 맞추는 제약은 구면(S²) 위 한 점을
// 고정하는 것과 같아서 2 DOF), 그 축을 중심으로 한 회전(roll) 1자유도가 안 잡혀서
// damped LSQ가 임의로 정하는 대로 남게 됨(대화 참고 - "비스듬하게" 되던 원인). 이
// 정렬을 추가로 같이 걸어서 그 남은 1자유도까지 마저 고정 - 두 제약을 동시에 만족하면
// orientation 전체(3 DOF)가 결정됨(3D 그래픽의 look-at + up-vector와 동일한 아이디어).
double eeYAxisAlignment(RobotModel& rm, const Eigen::VectorXd& q);

// torso + 왼팔(오른팔은 항상 고정, kRightArmIndices)을 사용해서 한 스텝 진행. 도달 가능한
// 거리에서 사용(scripts/lib/ik_solver.py의 ik_step()에 대응). 도달 불가능 모드 전용이었던
// ik_step_arm_only는 포팅하지 않음 — 그 기능은 곧 허리 고정 회전 방식으로 완전히 대체될 예정.
IkStepResult ikStep(RobotModel& rm, const Eigen::Vector3d& target_pos, const Eigen::VectorXd& q,
                     double damping, const IkStepParams& params,
                     const CollisionCheckFn& collision_checker = nullptr);

// eeDownAlignment/eeXAxisUpAlignment/eeYAxisAlignment처럼 (RobotModel&, q) -> 스칼라
// 정렬 스코어를 돌려주는 함수. orientationAlignStep이 이 타입으로 어떤 정렬 스코어든
// 받아서 풂.
using AlignmentScoreFn = std::function<double(RobotModel&, const Eigen::VectorXd&)>;

// orientationAlignStep에 넘기는 정렬 제약 하나 - score_fn(q)가 target_score에
// 도달하도록 풂. 여러 개를 동시에 넘기면(예: eeXAxisUpAlignment + eeYAxisAlignment)
// 그만큼의 스칼라 방정식을 한꺼번에 풀어서 orientation의 여러 자유도를 같이 고정함.
struct AlignmentConstraint {
  AlignmentScoreFn score_fn;
  double target_score;
};

// constraints의 각 score_fn(q)이 각자의 target_score에 도달하도록 torso_yaw + 왼팔
// 3관절만으로 한 스텝 진행. ikStep과 완전히 같은 구조(damped least squares +
// trust-region accept/reject)를 3D 위치 대신 N차원 스칼라 정렬 오차 벡터에 적용한
// 버전 - VIRTUAL_PLACE 그리퍼 오픈 시퀀스의 "놓는 자세" 정렬에 씀(fsm.cpp
// stepPlaceIk 참고). 이전엔 각 정렬 스코어의 dq를 매 tick 고정 크기로 그냥 더하기만
// 했는데, 그 방식은 threshold를 물리적으로 못 넘는 자세에서 dq가 계속 더해지며
// 진동했음(대화 참고) - damping이 rho(선형근사 신뢰도)를 보고 스텝 크기를 자동으로
// 줄여서, 못 넘는 상황이면 damping이 상한까지 올라가며 자연히(진동 없이) 멈추게 함.
IkStepResult orientationAlignStep(RobotModel& rm,
                                   const std::vector<AlignmentConstraint>& constraints,
                                   const Eigen::VectorXd& q, double damping,
                                   const IkStepParams& params,
                                   const CollisionCheckFn& collision_checker = nullptr);

namespace detail {
// min ||A x - b||^2 s.t. lb <= x <= ub 를 active-set으로 품(box 제약만 지원, n<=8 정도의
// 작은 문제 전용). ik_step 내부 구현이지만 gtest에서 직접 검증하기 위해 여기 노출해둠 —
// 외부(라이브러리 사용자)가 직접 쓸 일은 없는 내부 헬퍼라는 뜻으로 detail 네임스페이스에 둠.
Eigen::VectorXd solveBoundedLsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                 const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                 int max_iterations = 50);
}  // namespace detail

}  // namespace ik
