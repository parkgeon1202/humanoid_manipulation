// scripts/debug_ik_node.py의 C++ 포팅. ik_solve_node.py의 IK 계산과
// debug_node.cpp의 모터 명령을 하나로 합친 단독 디버그 노드.
//
// Python 버전과의 차이:
//  - 원래 있던 's'(도달 불가능, ik_step_arm_only)는 포팅하지 않음 - 그 대신 새 설계로
//    's' 키를 다시 살림: 비전(터미널 입력) 목표를 받으면 먼저 허리(torso_yaw)를
//    ik.torso_target_rad(기본 1.5)까지 고정 회전시키고, /motor_status로 실제 도달을
//    확인한 다음(속도+위치 둘 다 안정화), q_[TORSO_JOINT_INDEX]에 반영된 그 상태를
//    시작점으로 삼아 자동으로 도달 가능 IK(예전 'd' 로직)를 이어서 품. torso_yaw는
//    fixed_indices가 아니라 IK가 계속 자유롭게 움직일 수 있는 관절이라, 여기서 정한
//    각도는 "시작점"일 뿐 최종 각도가 아님(IK가 필요하면 계속 돌림). 이 흐름
//    전체가 한 번의 's' 입력으로 이어서 실행되므로 별도의 'd' 키는 더 이상 없음.
//  - ik.max_damping_streak_ticks(예전 도달 불가능 모드 전용 파라미터)는 포팅 안 함.
//
// 비전 대신 터미널에서 x, y, z를 직접 입력받는 점, /dynamixel_control로 모터 명령을
// 직접 publish하는 점(중간에 FSM 노드 없음), 조종하지 않는 모터를 's' 모드 진입
// 시점의 /motor_status 스냅샷에 고정하는 점(_frozen_motor_positions)은 Python과 동일.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include "dynamixel_hardware_msgs/msg/current_motor_status.hpp"
#include "dynamixel_hardware_msgs/msg/dynamixel_control_msgs.hpp"
#include "ik/collision_model.hpp"
#include "ik/ik_solver.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "visualization_msgs/msg/marker.hpp"

using dynamixel_hardware_msgs::msg::CurrentMotorStatus;
using dynamixel_hardware_msgs::msg::DynamixelControlMsgs;
using dynamixel_hardware_msgs::msg::DynamixelMsgs;

namespace {

// debug_node.cpp의 kMotorIdOrder와 동일(같은 순서로 dynamixel_control/motor_status
// 배열이 채워짐). tilt(23번)는 이 시스템에서 아예 제어 안 해서 목록에서 뺌.
const std::vector<std::string> kMotorIdOrder = {
    "0", "1", "2", "3", "4", "5", "6", "10", "11", "12", "13", "14",
    "15", "16", "17", "18", "19", "20", "21", "22"};

// debug_node.cpp의 kJointNameToMotorId와 동일한 관계를 q 인덱스 기준으로 표현:
// torso_yaw=q[0]->22, left_shoulder_pitch=q[1]->0, left_shoulder_roll=q[2]->2,
// left_elbow_pitch=q[3]->4. 그리퍼는 q가 아니라 별도 상태(gripper_position_)로 관리.
const std::map<int, std::string> kQIndexToMotorId = {
    {ik::kTorsoJointIndex, "22"},
    {ik::kLeftArmIndices[0], "0"},
    {ik::kLeftArmIndices[1], "2"},
    {ik::kLeftArmIndices[2], "4"},
};
const std::string kGripperMotorId = "6";

std::string resolveUrdfPath() {
  return ament_index_cpp::get_package_share_directory("humanoid_manipulation") +
         "/model/irc_man.urdf";
}

// Python의 _get_key: raw 모드로 잠깐 들어가서 timeout 동안 문자 하나를 기다리고,
// 읽든 못 읽든 매번 원래(cooked) 모드로 복원함 - 그래서 이후 std::cin >> 같은
// 캐노니컬 입력이 문제 없이 정상 동작함(promptTarget 참고).
char getKey(const termios& original, double timeout_sec) {
  termios raw = original;
  raw.c_lflag &= ~(ICANON | ECHO);
  raw.c_cc[VMIN] = 0;
  raw.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &raw);

  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(STDIN_FILENO, &fds);
  timeval tv;
  tv.tv_sec = static_cast<time_t>(timeout_sec);
  tv.tv_usec = static_cast<suseconds_t>((timeout_sec - tv.tv_sec) * 1e6);

  char key = '\0';
  if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0) {
    if (read(STDIN_FILENO, &key, 1) != 1) key = '\0';
  }
  tcsetattr(STDIN_FILENO, TCSADRAIN, &original);
  return key;
}

std::optional<std::array<double, 3>> promptTarget(char key) {
  std::cout << "\n[" << key << "] 목표 좌표 x y z (공백으로 구분): " << std::flush;
  double x, y, z;
  if (!(std::cin >> x >> y >> z)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "잘못된 입력이라 무시함(예: 0.3 0.1 0.2 처럼 입력).\n";
    return std::nullopt;
  }
  return std::array<double, 3>{x, y, z};
}

}  // namespace

class DebugIkNode : public rclcpp::Node {
 public:
  enum class Mode { kIdle, kRotatingTorso, kReachable };

  DebugIkNode() : rclcpp::Node("debug_ik_node"), rm_(resolveUrdfPath()), collision_checker_(rm_) {
    declareAndReadParameters();

    auto motor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    control_pub_ = create_publisher<DynamixelControlMsgs>(control_topic_, motor_qos);
    joint_state_pub_ =
        create_publisher<sensor_msgs::msg::JointState>("joint_states", rclcpp::QoS(10));
    target_marker_pub_ =
        create_publisher<visualization_msgs::msg::Marker>("/target_marker", rclcpp::QoS(10));
    status_sub_ = create_subscription<CurrentMotorStatus>(
        status_topic_, motor_qos,
        [this](const CurrentMotorStatus::SharedPtr msg) { onMotorStatus(msg); });

    timer_ = create_wall_timer(std::chrono::duration<double>(step_period_sec_),
                                [this]() { onTick(); });
  }

  Mode mode() const { return mode_.load(); }

  void requestKey(char key) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_key_ = key;
    pending_target_.reset();
  }

  void requestTarget(char key, double x, double y, double z) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_key_ = key;
    pending_target_ = Eigen::Vector3d(x, y, z);
  }

 private:
  void declareAndReadParameters() {
    control_topic_ = declare_parameter<std::string>("control_topic", "/dynamixel_control");
    status_topic_ = declare_parameter<std::string>("status_topic", "/motor_status");
    step_period_sec_ = declare_parameter<double>("ik.step_period_sec", 0.02);
    default_damping_ = declare_parameter<double>("ik.default_damping", ik::kDefaultDamping);
    min_damping_ = declare_parameter<double>("ik.min_damping", ik::kMinDamping);
    max_damping_ = declare_parameter<double>("ik.max_damping", ik::kMaxDamping);
    damping_decrease_factor_ =
        declare_parameter<double>("ik.damping_decrease_factor", ik::kDampingDecreaseFactor);
    damping_increase_factor_ =
        declare_parameter<double>("ik.damping_increase_factor", ik::kDampingIncreaseFactor);
    ik_tol_ = declare_parameter<double>("ik.tol", 1e-4);
    ik_alpha_ = declare_parameter<double>("ik.alpha", 0.5);
    trust_region_good_ratio_ =
        declare_parameter<double>("ik.trust_region_good_ratio", ik::kTrustRegionGoodRatio);
    trust_region_acceptable_ratio_ = declare_parameter<double>(
        "ik.trust_region_acceptable_ratio", ik::kTrustRegionAcceptableRatio);
    const double gripper_open_deg = declare_parameter<double>("gripper.open_deg", 30.0);
    const double gripper_closed_deg = declare_parameter<double>("gripper.closed_deg", 0.0);
    const double gripper_step_deg = declare_parameter<double>("gripper.step_deg", 0.6);
    gripper_open_rad_ = gripper_open_deg * M_PI / 180.0;
    gripper_closed_rad_ = gripper_closed_deg * M_PI / 180.0;
    gripper_step_rad_ = gripper_step_deg * M_PI / 180.0;
    // [torso_yaw, left_shoulder_pitch, left_shoulder_roll, left_elbow_pitch] -
    // left_shoulder_roll 기본값은 main_node.yaml의 ik.left_shoulder_roll_target_rad와
    // 동일하게 1.0.
    const std::vector<double> home_q = declare_parameter<std::vector<double>>(
        "pick_ready_home_q", std::vector<double>{0.0, 0.0, 1.0, 0.0});
    home_profile_velocity_ = declare_parameter<double>("profile_velocity", 1.0);
    const std::vector<double> joint_weights_4 = declare_parameter<std::vector<double>>(
        "ik.joint_weights", std::vector<double>{1.0, 1.0, 1.0, 1.0});
    joint_weight_scale_ = declare_parameter<double>("ik.joint_weight_scale", 0.0);
    secondary_gain_ = declare_parameter<double>("ik.secondary_gain", 0.0);
    collision_avoidance_threshold_m_ =
        declare_parameter<double>("ik.collision_avoidance_threshold_m", 0.03);
    collision_avoidance_k_pull_ = declare_parameter<double>("ik.collision_avoidance_k_pull", 1.0);
    stuck_streak_ticks_ = declare_parameter<int>("ik.stuck_streak_ticks", 30);
    torso_kick_offsets_ = declare_parameter<std::vector<double>>(
        "ik.torso_kick_offsets", std::vector<double>{0.5, 1.0, 1.5, -0.5, -1.0, -1.5});

    // 's' 모드 1단계(허리 고정 회전) 전용. torso_target_rad까지 회전시킨 다음,
    // 모터 속도가 torso_settle_velocity_threshold 미만이고 동시에 목표와의 위치
    // 차이가 torso_settle_position_tolerance_rad(rad) 이내면 "도달"로 보고 2단계
    // (도달 가능 IK)로 자동 전환함. 속도만 보면 명령 직후(아직 움직이기 전) 첫
    // 틱에 오판할 수 있어서 위치 조건을 반드시 같이 봄. CurrentMotorStatus.velocity는
    // rad/s가 아니라 raw uint8 단위라, debug_node.cpp의 arm_settled_velocity_threshold
    // 기본값(0.3)과 동일한 스케일을 그대로 씀.
    torso_target_rad_ = declare_parameter<double>("ik.torso_target_rad", -1.5);
    torso_settle_velocity_threshold_ =
        declare_parameter<double>("ik.torso_settle_velocity_threshold", 0.3);
    torso_settle_position_tolerance_rad_ =
        declare_parameter<double>("ik.torso_settle_position_tolerance_rad", 0.3);

    // home_q_full_: a키(publishHome)로 돌아갈 home 자세 전체(model.nq=8) 미리 계산해둠
    // (torso_yaw + 왼팔만 채우고 나머지 0). joint_weights_도 동일한 패턴: 실제로
    // 넘어오는 4개(torso_yaw + 왼팔)만 채우고, 오른팔/tilt는 fixed_indices라 값이
    // 의미 없으므로 중립값 1.0으로 둠.
    const int nq = rm_.model().nq;
    home_q_full_ = Eigen::VectorXd::Zero(nq);
    joint_weights_ = Eigen::VectorXd::Ones(nq);
    if (home_q.size() >= 4) {
      home_q_full_[ik::kTorsoJointIndex] = home_q[0];
      for (int i = 0; i < 3; ++i) home_q_full_[ik::kLeftArmIndices[i]] = home_q[i + 1];
    }
    if (joint_weights_4.size() >= 4) {
      joint_weights_[ik::kTorsoJointIndex] = joint_weights_4[0];
      for (int i = 0; i < 3; ++i) joint_weights_[ik::kLeftArmIndices[i]] = joint_weights_4[i + 1];
    }

    q_ = Eigen::VectorXd::Zero(nq);
    gripper_position_ = gripper_open_rad_;
    damping_ = default_damping_;
  }

  void onMotorStatus(const CurrentMotorStatus::SharedPtr msg) {
    const size_t n = std::min(kMotorIdOrder.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i) {
      motor_positions_[kMotorIdOrder[i]] = msg->position[i];
    }
    const size_t nv = std::min(kMotorIdOrder.size(), msg->velocity.size());
    for (size_t i = 0; i < nv; ++i) {
      motor_velocities_[kMotorIdOrder[i]] = msg->velocity[i];
    }
  }

  void consumePendingKey() {
    char key = '\0';
    std::optional<Eigen::Vector3d> target;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      key = pending_key_;
      pending_key_ = '\0';
      target = pending_target_;
      pending_target_.reset();
    }
    if (key == '\0') return;

    if (key == 'a') {
      mode_.store(Mode::kIdle);
      publishHome();
    } else if (key == 's' && target.has_value()) {
      mode_.store(Mode::kRotatingTorso);
      target_pos_ = *target;
      ik_converged_ = false;
      damping_ = default_damping_;
      stuck_streak_ = 0;
      torso_kick_index_ = 0;
      kick_settling_ = false;
      done_printed_ = false;
      gripper_position_ = gripper_open_rad_;
      frozen_motor_positions_ = motor_positions_;
      // torso만 목표 각도로 두고 나머지(왼팔)는 0에서 새로 시작 - 이전 시도의
      // 중간 상태가 남아있지 않게 함(reachable IK의 stuck-streak 리셋과 동일 이유).
      q_ = Eigen::VectorXd::Zero(rm_.model().nq);
      q_[ik::kTorsoJointIndex] = torso_target_rad_;
      q_[ik::kLeftArmIndices[1]] = 1.0;
      RCLCPP_INFO(get_logger(), "허리 %.2frad 회전 시작.", torso_target_rad_);
    }
  }

  void onTick() {
    consumePendingKey();
    const Mode current_mode = mode_.load();
    if (current_mode == Mode::kRotatingTorso) {
      stepRotatingTorso();
    } else if (current_mode == Mode::kReachable) {
      stepReachable();
    }
  }

  void stepRotatingTorso() {
    // 실제 이동은 모터 자체의 profile_velocity 기반 궤적 생성이 담당함(다른
    // 모드들과 동일 패턴) - 여기서는 매 틱 같은 목표를 계속 publish하면서
    // motor_status 피드백으로 도달 여부만 확인함.
    publishArm(home_profile_velocity_);

    const std::string& torso_motor_id = kQIndexToMotorId.at(ik::kTorsoJointIndex);
    const std::string& shoulder_roll_motor_id = kQIndexToMotorId.at(ik::kLeftArmIndices[1]);
    const auto torso_pos_it = motor_positions_.find(torso_motor_id);
    const auto torso_vel_it = motor_velocities_.find(torso_motor_id);
    const auto roll_pos_it = motor_positions_.find(shoulder_roll_motor_id);
    const auto roll_vel_it = motor_velocities_.find(shoulder_roll_motor_id);
    if (torso_pos_it == motor_positions_.end() || torso_vel_it == motor_velocities_.end() ||
        roll_pos_it == motor_positions_.end() || roll_vel_it == motor_velocities_.end()) {
      return;
    }

    const bool torso_settled =
        std::fabs(torso_vel_it->second) < torso_settle_velocity_threshold_ &&
        std::fabs(torso_pos_it->second - torso_target_rad_) < torso_settle_position_tolerance_rad_;
    const bool roll_settled =
        std::fabs(roll_vel_it->second) < torso_settle_velocity_threshold_ &&
        std::fabs(roll_pos_it->second - 1.0) < torso_settle_position_tolerance_rad_;
    const bool settled = torso_settled && roll_settled;
    if (settled) {
      mode_.store(Mode::kReachable);
      RCLCPP_INFO(get_logger(), "허리 회전 완료(%.2frad) -> 도달 가능 IK 시작.",
                  torso_target_rad_);
    }
  }

  Eigen::VectorXd secondaryDq() {
    return collision_checker_.elbowTorsoAvoidanceDq(q_, collision_avoidance_threshold_m_,
                                                      collision_avoidance_k_pull_);
  }

  double effectiveSecondaryGain(const Eigen::VectorXd& secondary_dq) const {
    // secondary_dq가 정확히 0벡터면 secondary_gain을 켜둬도 "dq를 0으로 당기는" 순수
    // 추가 damping이 되어 일반 목표에서도 수렴이 막힘 - 진짜 뭔가를 원할 때만 켬.
    if (secondary_dq.squaredNorm() == 0.0) return 0.0;
    return secondary_gain_;
  }

  // 킥 직후 q_가 명령한 자세(torso+왼팔)에 모터가 실제로 도달했는지(위치+속도 둘 다
  // settle) 확인 - stepRotatingTorso의 도달 판정과 동일한 임계값을 재사용.
  bool armMotorsSettledToQ() const {
    for (const auto& kv : kQIndexToMotorId) {
      const auto pos_it = motor_positions_.find(kv.second);
      const auto vel_it = motor_velocities_.find(kv.second);
      if (pos_it == motor_positions_.end() || vel_it == motor_velocities_.end()) return false;
      if (std::fabs(vel_it->second) >= torso_settle_velocity_threshold_) return false;
      if (std::fabs(pos_it->second - q_[kv.first]) >= torso_settle_position_tolerance_rad_) {
        return false;
      }
    }
    return true;
  }

  void stepReachable() {
    if (!ik_converged_) {
      if (kick_settling_) {
        // 킥으로 리셋한 q_를 시작점으로 IK를 바로 이어가지 않고, 모터가 실제로
        // 그 자세에 도달할 때까지 대기(publishArm은 아래에서 계속 나가서 모터는
        // 계속 그 목표로 움직임).
        if (armMotorsSettledToQ()) {
          kick_settling_ = false;
        }
        publishArm(home_profile_velocity_);
        return;
      }
      const Eigen::VectorXd secondary_dq = secondaryDq();

      ik::IkStepParams params;
      params.tol = ik_tol_;
      params.alpha = ik_alpha_;
      params.min_damping = min_damping_;
      params.max_damping = max_damping_;
      params.damping_decrease_factor = damping_decrease_factor_;
      params.damping_increase_factor = damping_increase_factor_;
      params.joint_weights = joint_weights_;
      params.joint_weight_scale = joint_weight_scale_;
      params.secondary_dq = secondary_dq;
      params.secondary_gain = effectiveSecondaryGain(secondary_dq);
      params.trust_region_good_ratio = trust_region_good_ratio_;
      params.trust_region_acceptable_ratio = trust_region_acceptable_ratio_;

      const ik::CollisionCheckFn collision_fn = [this](const Eigen::VectorXd& q_candidate) {
        return collision_checker_.isColliding(q_candidate);
      };
      const ik::IkStepResult result =
          ik::ikStep(rm_, target_pos_, q_, damping_, params, collision_fn);
      q_ = result.q;
      damping_ = result.damping;

      // damping이 상한에 붙어있어도 rho(선형근사 불신) 때문이면 "막힘"이 아닐 수
      // 있음 - hard_rejected(진짜 충돌 반려)만 카운트해서 진짜 정체만 킥이 발동하게 함.
      if (result.hard_rejected) {
        ++stuck_streak_;
      } else {
        stuck_streak_ = std::max(0, stuck_streak_ - 1);
      }
      if (stuck_streak_ >= stuck_streak_ticks_) {
        const double kick = torso_kick_offsets_[torso_kick_index_ % torso_kick_offsets_.size()];
        ++torso_kick_index_;
        q_ = Eigen::VectorXd::Zero(rm_.model().nq);
        q_[ik::kTorsoJointIndex] = kick;
        q_[ik::kLeftArmIndices[1]] = 1.0;
        damping_ = default_damping_;
        stuck_streak_ = 0;
        kick_settling_ = true;
        RCLCPP_INFO(get_logger(),
                    "reachable IK 정체 감지 -> 전체 관절 리셋 + torso_yaw=%.2frad로 재시도.",
                    kick);
      }

      if (result.converged) {
        ik_converged_ = true;
        RCLCPP_INFO(get_logger(), "reachable IK converged, 그립 시작.");
      }
    } else {
      gripper_position_ = std::max(gripper_closed_rad_, gripper_position_ - gripper_step_rad_);
      if (gripper_position_ <= gripper_closed_rad_ && !done_printed_) {
        done_printed_ = true;
        RCLCPP_INFO(get_logger(), "푸는 거 다 끝났음.");
      }
    }
    publishArm(home_profile_velocity_);
  }

  void publishHome() {
    q_ = home_q_full_;
    publishArm(home_profile_velocity_);
  }

  void publishArm(double profile_velocity) {
    // 's' 모드(허리 회전 + 도달 가능 IK)로 진행 중일 때는 진입 시점 스냅샷을 계속
    // 쓰고, 그 외(home 등)에는 매번 최신 motor_status를 그대로 되돌려보냄.
    const std::map<std::string, double>& base_positions =
        (mode_.load() != Mode::kIdle) ? frozen_motor_positions_ : motor_positions_;
    std::map<std::string, double> goal_by_motor = base_positions;
    for (const auto& kv : kQIndexToMotorId) {
      goal_by_motor[kv.second] = q_[kv.first];
    }
    goal_by_motor[kGripperMotorId] = gripper_position_;

    DynamixelControlMsgs msg;
    const auto stamp = get_clock()->now();
    msg.motor_control.reserve(kMotorIdOrder.size());
    for (const auto& motor_id : kMotorIdOrder) {
      DynamixelMsgs motor_msg;
      motor_msg.header.stamp = stamp;
      const auto it = goal_by_motor.find(motor_id);
      motor_msg.goal_position =
          static_cast<float>(it != goal_by_motor.end() ? it->second : 0.0);
      motor_msg.profile_acceleration = 0.0f;
      motor_msg.profile_velocity = static_cast<float>(profile_velocity);
      msg.motor_control.push_back(motor_msg);
    }
    control_pub_->publish(msg);

    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    js.name = {"torso_yaw",           "left_shoulder_pitch",  "left_shoulder_roll",
               "left_elbow_pitch",    "right_shoulder_pitch", "right_shoulder_roll",
               "right_elbow_pitch",   "tilt"};
    js.position.assign(q_.data(), q_.data() + q_.size());
    joint_state_pub_->publish(js);

    if (mode_.load() != Mode::kIdle) {
      publishTargetMarker(stamp);
    }
  }

  void publishTargetMarker(const rclcpp::Time& stamp) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = stamp;
    marker.ns = "debug_ik_target";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = target_pos_.x();
    marker.pose.position.y = target_pos_.y();
    marker.pose.position.z = target_pos_.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.02;
    marker.color.r = 1.0f;
    marker.color.a = 1.0f;
    target_marker_pub_->publish(marker);
  }

  ik::RobotModel rm_;
  ik::CollisionChecker collision_checker_;

  std::string control_topic_;
  std::string status_topic_;
  double step_period_sec_ = 0.02;
  double default_damping_ = ik::kDefaultDamping;
  double min_damping_ = ik::kMinDamping;
  double max_damping_ = ik::kMaxDamping;
  double damping_decrease_factor_ = ik::kDampingDecreaseFactor;
  double damping_increase_factor_ = ik::kDampingIncreaseFactor;
  double ik_tol_ = 1e-4;
  double ik_alpha_ = 0.5;
  double trust_region_good_ratio_ = ik::kTrustRegionGoodRatio;
  double trust_region_acceptable_ratio_ = ik::kTrustRegionAcceptableRatio;
  double gripper_open_rad_ = 0.0;
  double gripper_closed_rad_ = 0.0;
  double gripper_step_rad_ = 0.0;
  double home_profile_velocity_ = 1.0;
  double secondary_gain_ = 0.0;
  double collision_avoidance_threshold_m_ = 0.03;
  double collision_avoidance_k_pull_ = 1.0;
  int stuck_streak_ticks_ = 30;
  std::vector<double> torso_kick_offsets_;
  Eigen::VectorXd joint_weights_;
  double joint_weight_scale_ = 0.0;
  Eigen::VectorXd home_q_full_;
  double torso_target_rad_ = -1.5;
  double torso_settle_velocity_threshold_ = 0.3;
  double torso_settle_position_tolerance_rad_ = 0.05;

  rclcpp::Publisher<DynamixelControlMsgs>::SharedPtr control_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_marker_pub_;
  rclcpp::Subscription<CurrentMotorStatus>::SharedPtr status_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mutex_;
  char pending_key_ = '\0';
  std::optional<Eigen::Vector3d> pending_target_;

  std::map<std::string, double> motor_positions_;
  std::map<std::string, double> motor_velocities_;
  std::map<std::string, double> frozen_motor_positions_;

  Eigen::VectorXd q_;
  double gripper_position_ = 0.0;
  std::atomic<Mode> mode_{Mode::kIdle};
  Eigen::Vector3d target_pos_ = Eigen::Vector3d::Zero();
  bool ik_converged_ = false;
  double damping_ = ik::kDefaultDamping;
  int stuck_streak_ = 0;
  size_t torso_kick_index_ = 0;
  bool kick_settling_ = false;
  bool done_printed_ = false;
};

namespace {

void keyboardLoop(const std::shared_ptr<DebugIkNode>& node) {
  termios original;
  tcgetattr(STDIN_FILENO, &original);
  std::cout << "a: home(IK 없음)  s: 허리 회전 후 도달 가능 IK + 그립(좌표 입력)  "
               "(Ctrl+C: 종료)\n";
  try {
    while (rclcpp::ok()) {
      const char key = getKey(original, 0.1);
      if (key == 'a') {
        node->requestKey('a');
      } else if (key == 's') {
        if (node->mode() != DebugIkNode::Mode::kIdle) {
          std::cout << "\n이미 다른 모드가 진행 중이라 무시함.\n";
          continue;
        }
        const auto target = promptTarget('s');
        if (target.has_value()) {
          node->requestTarget('s', (*target)[0], (*target)[1], (*target)[2]);
        }
      } else if (key == '\x03') {
        break;
      }
    }
  } catch (...) {
    tcsetattr(STDIN_FILENO, TCSADRAIN, &original);
    throw;
  }
  tcsetattr(STDIN_FILENO, TCSADRAIN, &original);
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DebugIkNode>();

  std::thread spin_thread([node]() { rclcpp::spin(node); });

  keyboardLoop(node);

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
