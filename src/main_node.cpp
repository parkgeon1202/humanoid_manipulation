// FSM + IK + 모터 I/O를 전부 담당하는 노드(구 main_node.cpp + ik_solve_node.py 흡수).
// 모든 상태/판정 로직은 fsm.cpp(ManipulationFSM) 안에 있고, 이 파일은 콜백에서 받은
// 값을 fsm_에 그대로 전달한 뒤 돌아온 FsmAction을 기계적으로 ROS 액션(publish/모션
// 재생/deactivate)으로 옮기기만 하는 얇은 글루임.
//
// IK 연산(solve_tick(), 느림)은 전용 콜백 그룹 + 전용 스레드에서 돌고, motor_status
// 등 메시지 콜백(빨라야 함)은 다른 스레드에서 돎(MultiThreadedExecutor) - 원래
// main_node.cpp/ik_solve_node.py를 별도 프로세스로 뒀던 이유와 같은 격리를 스레드
// 수준에서 재현함. fsm_에 실제 로직(on_activated/on_motor_feedback/on_motion_end/
// solve_tick)을 태우는 호출은 전부 on_solve_tick() 한 스레드에서만 하고, 메시지
// 콜백 스레드는 원시 데이터만 fsm_에 써넣음(그 setter들은 fsm.cpp 내부에서 자체
// 락으로 보호) - 그래서 이 파일은 별도 뮤텍스가 필요 없음.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "dynamixel_hardware_msgs/msg/current_motor_status.hpp"
#include "dynamixel_hardware_msgs/msg/dynamixel_control_msgs.hpp"
#include "dynamixel_hardware_msgs/msg/dynamixel_msgs.hpp"
#include "fsm.hpp"
#include "irc_humanoid_interfaces/msg/master2_vision_msg.hpp"
#include "irc_humanoid_interfaces/msg/motion_operator.hpp"
#include "irc_humanoid_interfaces/msg/vision2_mani_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker.hpp"

using CallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using dynamixel_hardware_msgs::msg::CurrentMotorStatus;
using dynamixel_hardware_msgs::msg::DynamixelControlMsgs;
using dynamixel_hardware_msgs::msg::DynamixelMsgs;
using irc_humanoid_interfaces::msg::Master2VisionMsg;
using irc_humanoid_interfaces::msg::MotionOperator;
using irc_humanoid_interfaces::msg::Vision2ManiMsg;
using rclcpp_lifecycle::State;
using std_msgs::msg::Bool;
using visualization_msgs::msg::Marker;

namespace
{
// motor_status(CurrentMotorStatus)/dynamixel_control(DynamixelControlMsgs)의
// 배열은 이 순서(모터 id)대로 채워짐 -> 인덱스 i가 kMotorIdOrder[i]에 대응.
// tilt(23번)는 이 시스템에서 아예 제어 안 해서 목록에서 뺌(fsm.cpp와 동일해야 함).
const std::vector<std::string> kMotorIdOrder = {
  "0", "1", "2", "3", "4", "5", "6",
  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
  "22"};

// 그립 모터(6번)는 이제 combined /dynamixel_control이 아니라 gripper_topic_로 단독
// 전송함(kMotorIdOrder에서 위치 파악용으로는 여전히 필요해서 목록 자체는 유지).
const std::string kGripMotorId = "6";

// /master/pan_tilt(Master2VisionMsg)에 보내는 place용 tilt 모드 번호.
constexpr int32_t kPlaceTiltMode = 5;

std::string resolveUrdfPath()
{
  return ament_index_cpp::get_package_share_directory("humanoid_manipulation") +
         "/model/irc_man.urdf";
}
}  // namespace

class ManipulationLifecycleNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  ManipulationLifecycleNode() : rclcpp_lifecycle::LifecycleNode("manipulation_node_cpp") {
    // 노드 프로세스가 뜨자마자 configure까지는 스스로 진행 (기존과 동일).
    this->configure();
  }

  CallbackReturn on_configure(const State &) override {
    declare_and_get_params();
    create_activate_cmd_sub();

    RCLCPP_INFO(get_logger(), "Configured. Waiting for activate_cmd.");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const State &) override {
    deactivate_pending_ = false;
    deferred_deactivate_timer_.reset();

    // 모터 명령/상태는 매 주기 계속 갱신되는 스트림이라 최신 값만 중요 ->
    // BEST_EFFORT + depth=1.
    auto motor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    control_pub_ = create_publisher<DynamixelControlMsgs>(control_topic_, motor_qos);
    gripper_pub_ = create_publisher<DynamixelMsgs>(gripper_topic_, motor_qos);
    // dynamixel_hardware_interface의 /RealsensePanTilt를 직접 건드리지 않고,
    // realsense_pan_tilt_node가 구독하는 /master/pan_tilt(Master2VisionMsg)로 모드
    // 번호를 보냄 - 그 노드가 모드->실제 각도 변환 + /RealsensePanTilt publish까지
    // 대신함. 구독 쪽이 rclcpp::QoS(10)(기본 RELIABLE)라 여기도 맞춰줌(motor_qos는
    // BEST_EFFORT라 그대로 쓰면 reliability가 안 맞아 디스커버리부터 연결이 안 됨).
    pan_tilt_pub_ = create_publisher<Master2VisionMsg>(
      pan_tilt_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());

    // RViz 디버그 시각화 전용(debug_ik_node.cpp의 /target_marker와 동일 토픽/규칙을
    // 재사용해서 같은 RViz 설정을 그대로 씀) - target_pos_는 큰 구, 활성 궤적
    // 샘플들은 작은 점들로 매 tick 갱신해서 publish함(on_solve_tick 참고).
    // display.rviz의 TargetMarker 디스플레이가 /target_marker를 RELIABLE로 구독하게
    // 돼 있음(depth=5) - 그대로 맞춰야 디스커버리부터 연결됨(motor_qos의 BEST_EFFORT를
    // 그대로 쓰면 pan_tilt 때와 같은 문제가 재발함).
    auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    target_marker_pub_ = create_publisher<Marker>("/target_marker", marker_qos);
    trajectory_marker_pub_ = create_publisher<Marker>("/trajectory_marker", marker_qos);
    status_sub_ = create_subscription<CurrentMotorStatus>(
      status_topic_, motor_qos,
      [this](const CurrentMotorStatus::SharedPtr msg) {on_motor_status(msg);});

    // 비전 노드가 보내는 원본 좌표(공/골대 x/y/z 둘 다 포함, mm 단위). master(task_planner)를
    // 거치지 않고 직접 구독함 - on_vision2mani()에서 activate_count_ 홀짝으로 이번 활성화
    // 사이클이 PICK(공)인지 PLACE(골대)인지 판단해서 그 값만 update_vision에 넘김.
    ++activate_count_;
    vision_sub_ = create_subscription<Vision2ManiMsg>(
      "/vision2mani", rclcpp::QoS(10),
      [this](const Vision2ManiMsg::SharedPtr msg) {on_vision2mani(msg);});

    // 모션 재생 요청/완료 통지는 유실되면 안 되는 이산적 이벤트 -> RELIABLE.
    auto motion_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    motion_operator_pub_ = create_publisher<MotionOperator>("motion_operator", motion_qos);
    motion_end_sub_ = create_subscription<MotionOperator>(
      "motion_end", motion_qos,
      [this](const MotionOperator::SharedPtr msg) {on_motion_end(msg);});

    // solve_timer_(아래)가 생기기 전에 반드시 먼저 불러야 함 - on_activated()는
    // fsm_ 로직 호출이라 solve_tick 스레드 하나에서만 돌아야 하는데, 타이머가 이미
    // 있는 상태에서 부르면 그 첫 tick과 레이스가 날 수 있음(타이머가 아예 없는
    // 시점에 부르면 원천적으로 불가능).
    apply_action(fsm_->on_activated());

    // IK 연산(solve_tick())은 전용 콜백 그룹에 묶어서 motor_status 등 메시지
    // 콜백과 별도 스레드에서 돌게 함(main()의 MultiThreadedExecutor 참고).
    solve_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    solve_timer_ = create_wall_timer(
      std::chrono::duration<double>(step_period_sec_),
      [this]() {on_solve_tick();},
      solve_callback_group_);

    RCLCPP_INFO(
      get_logger(), "Activated. Listening on %s / %s.",
      control_topic_.c_str(), status_topic_.c_str());
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const State &) override {
    // COMPLETE_PLACE 종료는 solve_timer_ 콜백 안에서 deactivate()를
    // 호출한다. 그 시점에 현재 실행 중인 타이머를 reset()하면
    // 콜백 객체가 해제되어 segfault가 날 수 있으므로 스케줄링만 취소한다.
    // 다음 on_activate()의 대입 시점에는 이 콜백이 종료된 후라 안전하다.
    if (solve_timer_) {
      solve_timer_->cancel();
    }
    control_pub_.reset();
    gripper_pub_.reset();
    pan_tilt_pub_.reset();
    target_marker_pub_.reset();
    trajectory_marker_pub_.reset();
    status_sub_.reset();
    vision_sub_.reset();
    motion_operator_pub_.reset();
    motion_end_sub_.reset();

    // 정상적으로 PLACE까지 끝난 경우에만 다음 공을 처리할 수 있도록 PICK 처음으로
    // 되돌린다. 외부에서 activate_cmd(false)를 보내 작업 중간에 비활성화한 경우에는
    // 진행 상태를 보존해 재활성화 시 이어서 수행한다.
    if (fsm_->is_sequence_complete()) {
      fsm_->reset();
      RCLCPP_INFO(get_logger(), "Sequence complete. FSM reset for the next pick/place cycle.");
    }

    RCLCPP_INFO(get_logger(), "Deactivated. Waiting for activate_cmd.");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_cleanup(const State &) override {
    activate_cmd_sub_.reset();
    activate_end_pub_.reset();
    deferred_deactivate_timer_.reset();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_shutdown(const State &) override {
    return CallbackReturn::SUCCESS;
  }

private:
  void declare_and_get_params() {
    const std::string urdf_path =
      declare_parameter<std::string>("urdf_path", resolveUrdfPath());
    const std::string ee_frame = declare_parameter<std::string>("ee_frame", "ee_link_1");
    fsm_ = std::make_unique<ManipulationFSM>(urdf_path, ee_frame);

    control_topic_ = declare_parameter<std::string>("control_topic", "/dynamixel_control");
    gripper_topic_ = declare_parameter<std::string>("gripper_topic", "/gripper_control");
    status_topic_ = declare_parameter<std::string>("status_topic", "/motor_status");
    pan_tilt_topic_ = declare_parameter<std::string>("pan_tilt_topic", "/master/pan_tilt");
    step_period_sec_ = declare_parameter<double>("ik.step_period_sec", 0.02);

    IkTuning tuning;
    tuning.place_traj_dt_sec = step_period_sec_;
    tuning.place_apex_x = declare_parameter<double>("place.apex_x", 0.1);
    tuning.place_apex_y = declare_parameter<double>("place.apex_y", 0.3);
    tuning.place_traj_duration_sec = declare_parameter<double>("place.traj_duration_sec", 3.0);
    tuning.place_tol_increase_step_m =
      declare_parameter<double>("place.tol_increase_step_m", 0.01);
    tuning.place_tol_decrease_step_m =
      declare_parameter<double>("place.tol_decrease_step_m", 0.01);
    tuning.place_release_max_error_m =
      declare_parameter<double>("place.release_max_error_m", 0.02);
    tuning.pick_approach_duration_sec =
      declare_parameter<double>("pick.approach_duration_sec", 1.0);
    // complete_grip_lift_z_m 필드 주석 참고 - 집은 직후 EE를 z+로 들어올려서
    // 안전하게 들고 있는 자세로 IK를 재수렴시키는 거리(m).
    tuning.complete_grip_lift_z_m =
      declare_parameter<double>("pick.complete_grip_lift_z_m", 0.1);
    // complete_grip_settle_*_threshold 필드 주석 참고 - COMPLETE_GRIP z+lift 목표
    // settle 대기 전용(다른 곳의 허리/어깨 회전 대기엔 영향 없음).
    tuning.complete_grip_settle_velocity_threshold =
      declare_parameter<double>("pick.complete_grip_settle_velocity_threshold", 0.6);
    tuning.complete_grip_settle_position_tolerance_rad =
      declare_parameter<double>("pick.complete_grip_settle_position_tolerance_rad", 0.6);
    tuning.default_damping = declare_parameter<double>("ik.default_damping", ik::kDefaultDamping);
    tuning.min_damping = declare_parameter<double>("ik.min_damping", ik::kMinDamping);
    tuning.max_damping = declare_parameter<double>("ik.max_damping", ik::kMaxDamping);
    tuning.damping_decrease_factor =
      declare_parameter<double>("ik.damping_decrease_factor", ik::kDampingDecreaseFactor);
    tuning.damping_increase_factor =
      declare_parameter<double>("ik.damping_increase_factor", ik::kDampingIncreaseFactor);
    tuning.tol = declare_parameter<double>("ik.tol", 1e-4);
    tuning.alpha = declare_parameter<double>("ik.alpha", 0.5);
    tuning.trust_region_good_ratio =
      declare_parameter<double>("ik.trust_region_good_ratio", ik::kTrustRegionGoodRatio);
    tuning.trust_region_acceptable_ratio = declare_parameter<double>(
      "ik.trust_region_acceptable_ratio", ik::kTrustRegionAcceptableRatio);

    const double gripper_open_deg = declare_parameter<double>("gripper.open_deg", 30.0);
    const double gripper_closed_deg = declare_parameter<double>("gripper.closed_deg", -36.7033);
    // place_open_deg 필드 주석 참고 - VIRTUAL_PLACE에서 놓을 때 전용, gripper.open_deg
    // (PICK_READY 진입용)보다 더 크게 벌려야 할 수 있어서 따로 뺌.
    const double place_gripper_open_deg =
      declare_parameter<double>("gripper.place_open_deg", 60.0);
    // place_release_shoulder_pitch_offset_rad 필드 주석 참고 - 놓을 때 그리퍼 모터
    // 프레임이 아래로 가버리는 문제 대응용 어깨 pitch 추가 회전(도).
    const double place_release_shoulder_pitch_offset_deg = declare_parameter<double>(
      "gripper.place_release_shoulder_pitch_offset_deg", -50.0);
    // place_release_elbow_offset_rad 필드 주석 참고 - 같은 시점에 같이 도는 팔꿈치
    // 추가 회전(도).
    const double place_release_elbow_offset_deg =
      declare_parameter<double>("gripper.place_release_elbow_offset_deg", 30.0);
    tuning.gripper_open_rad = gripper_open_deg * M_PI / 180.0;
    tuning.gripper_closed_rad = gripper_closed_deg * M_PI / 180.0;
    tuning.place_gripper_open_rad = place_gripper_open_deg * M_PI / 180.0;
    tuning.place_release_shoulder_pitch_offset_rad =
      place_release_shoulder_pitch_offset_deg * M_PI / 180.0;
    tuning.place_release_elbow_offset_rad = place_release_elbow_offset_deg * M_PI / 180.0;

    // [torso_yaw, left_shoulder_pitch, left_shoulder_roll, left_elbow_pitch] -
    // left_shoulder_roll 기본값은 ik.left_shoulder_roll_target_rad와 동일하게 0.3.
    const std::vector<double> home_q = declare_parameter<std::vector<double>>(
      "pick_ready_home_q", std::vector<double>{0.0, -2.7, 0.3, 0.0});
    const std::vector<double> joint_weights_4 = declare_parameter<std::vector<double>>(
      "ik.joint_weights", std::vector<double>{1.0, 1.0, 1.0, 1.0});
    const std::vector<double> place_joint_weights_4 = declare_parameter<std::vector<double>>(
      "ik.place_joint_weights", std::vector<double>{10.0, 1.0, 1.0, 1.0});
    const std::vector<double> complete_place_joint_weights_4 = declare_parameter<std::vector<double>>(
      "ik.complete_place_joint_weights", std::vector<double>{100.0, 1.0, 1.0, 1.0});
    tuning.joint_weight_scale = declare_parameter<double>("ik.joint_weight_scale", 0.0);
    tuning.secondary_gain = declare_parameter<double>("ik.secondary_gain", 1.0);
    tuning.orientation_align_k_pull =
      declare_parameter<double>("ik.orientation_align_k_pull", 1.0);
    tuning.collision_avoidance_threshold_m =
      declare_parameter<double>("ik.collision_avoidance_threshold_m", 0.03);
    tuning.collision_avoidance_k_pull =
      declare_parameter<double>("ik.collision_avoidance_k_pull", 1.0);
    tuning.stuck_streak_ticks = declare_parameter<int>("ik.stuck_streak_ticks", 30);
    tuning.torso_kick_step_rad = declare_parameter<double>("ik.torso_kick_step_rad", 0.5);
    tuning.elbow_kick_rad = declare_parameter<double>("ik.elbow_kick_rad", 0.0);
    tuning.left_shoulder_pitch_kick_rad =
      declare_parameter<double>("ik.left_shoulder_pitch_kick_rad", -2.7);

    tuning.left_shoulder_roll_target_rad =
      declare_parameter<double>("ik.left_shoulder_roll_target_rad", 0.3);
    tuning.pick_ready_torso_bias_rad =
      declare_parameter<double>("ik.pick_ready_torso_bias_rad", 1.0);
    tuning.vision_y_offset = declare_parameter<double>("ik.vision_y_offset", 0.0328);
    tuning.settle_velocity_threshold =
      declare_parameter<double>("ik.torso_settle_velocity_threshold", 0.3);
    tuning.settle_position_tolerance_rad =
      declare_parameter<double>("ik.torso_settle_position_tolerance_rad", 0.05);

    tuning.grip_moving_velocity_threshold =
      declare_parameter<double>("grip.moving_velocity_threshold", 0.5);
    tuning.grip_near_zero_position_threshold =
      declare_parameter<double>("grip.near_zero_position_threshold", 0.1);
    tuning.grip_wiggle_kick_position_rad =
      declare_parameter<double>("grip.wiggle_kick_position_rad", 1.0);
    tuning.grip_stopped_streak_ticks =
      declare_parameter<int>("grip.stopped_streak_ticks", 10);
    tuning.grip_wiggle_restore_timeout_ticks =
      declare_parameter<int>("grip.wiggle_restore_timeout_ticks", 200);

    // pick_ready_home_q/joint_weights는 실제로 명령되는 4개(torso_yaw + 왼팔)만
    // 받고, 오른팔/tilt는 model.nq(8) 크기로 패딩함(debug_ik_node.cpp와 동일 패턴).
    const int nq = 8;
    tuning.home_q_full = Eigen::VectorXd::Zero(nq);
    tuning.joint_weights = Eigen::VectorXd::Ones(nq);
    tuning.place_joint_weights = Eigen::VectorXd::Ones(nq);
    tuning.complete_place_joint_weights = Eigen::VectorXd::Ones(nq);
    if (home_q.size() >= 4) {
      tuning.home_q_full[ik::kTorsoJointIndex] = home_q[0];
      for (int i = 0; i < 3; ++i) tuning.home_q_full[ik::kLeftArmIndices[i]] = home_q[i + 1];
    }
    if (joint_weights_4.size() >= 4) {
      tuning.joint_weights[ik::kTorsoJointIndex] = joint_weights_4[0];
      for (int i = 0; i < 3; ++i) tuning.joint_weights[ik::kLeftArmIndices[i]] = joint_weights_4[i + 1];
    }
    if (place_joint_weights_4.size() >= 4) {
      tuning.place_joint_weights[ik::kTorsoJointIndex] = place_joint_weights_4[0];
      for (int i = 0; i < 3; ++i) {
        tuning.place_joint_weights[ik::kLeftArmIndices[i]] = place_joint_weights_4[i + 1];
      }
    }
    if (complete_place_joint_weights_4.size() >= 4) {
      tuning.complete_place_joint_weights[ik::kTorsoJointIndex] = complete_place_joint_weights_4[0];
      for (int i = 0; i < 3; ++i) {
        tuning.complete_place_joint_weights[ik::kLeftArmIndices[i]] =
          complete_place_joint_weights_4[i + 1];
      }
    }

    fsm_->configure_ik(tuning);
  }

  void create_activate_cmd_sub() {
    // task_planner의 활성화 명령은 inactive 상태에서도 반드시 받아야 하므로
    // 유실 허용 안 됨 -> RELIABLE.
    auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    activate_cmd_sub_ = create_subscription<Bool>(
      "activate_cmd", cmd_qos,
      [this](const Bool::SharedPtr msg) {on_activate_cmd(msg);});

    // 픽/플레이스 시퀀스(DONE)가 끝나면 task_planner가 구독하는
    // 종료 전용 토픽으로 통지한다. 시작 명령과 응답을 분리해 자기 수신을 막는다.
    activate_end_pub_ = create_publisher<Bool>("activate_end", cmd_qos);
  }

  void on_activate_cmd(const Bool::SharedPtr msg) {
    // inactive/active 상태에서 언제든 들어올 수 있어 두 트리거를 그냥 시도하고,
    // 상태가 안 맞아 실패하면 조용히 무시.
    if (msg->data) {
      this->activate();
    } else {
      this->deactivate();
    }
  }

  // 데이터만 fsm_에 써넣음(내부적으로 자체 락 보호) - on_motor_feedback()은 더
  // 이상 여기서 안 부르고 on_solve_tick()이 매 tick 부름.
  void on_motor_status(const CurrentMotorStatus::SharedPtr msg) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "motor_status received: %zu motor(s)", msg->position.size());

    const size_t n = std::min(kMotorIdOrder.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i) {
      fsm_->set_motor_position(kMotorIdOrder[i], msg->position[i]);
    }
    const size_t nv = std::min(kMotorIdOrder.size(), msg->velocity.size());
    for (size_t i = 0; i < nv; ++i) {
      fsm_->set_motor_velocity(kMotorIdOrder[i], msg->velocity[i]);
    }

    // tilt(23번)는 kMotorIdOrder에 없어서(제어 안 함) 위 루프에서 안 채워짐 - 배열
    // 맨 끝이 tilt 실측값이라 따로 빼서 fsm_에 넣어줌(VIRTUAL_PLACE 비전 캡처가
    // isTiltSettled()로 카메라가 레벨까지 실제로 돌아왔는지 확인하는 용도).
    if (!msg->position.empty()) {
      fsm_->set_tilt_position(msg->position.back());
    }
    if (!msg->velocity.empty()) {
      fsm_->set_tilt_velocity(msg->velocity.back());
    }
  }

  // activate_count_는 on_activate()에서 매 활성화마다 1씩 늘어남 - 홀수 번째
  // 활성화(1, 3, 5, ...)는 SIT에서 시작하는 PICK 사이클이라 공(ball_cam) 좌표가
  // 필요하고, 짝수 번째(2, 4, 6, ...)는 DONE에서 재활성화되는 PLACE 사이클이라
  // 골대(goal_post_cam) 좌표가 필요함(fsm.cpp의 onActivatedImpl 주석 참고).
  // Vision2ManiMsg는 float64(mm 단위)라 1000.0으로 나누기만 하면 double 그대로 씀.
  void on_vision2mani(const Vision2ManiMsg::SharedPtr msg) {
    const bool is_pick_cycle = (activate_count_ % 2) == 1;
    const double x = (is_pick_cycle ? msg->ball_cam_x : msg->goal_post_cam_x) / 1000.0;
    const double y = (is_pick_cycle ? msg->ball_cam_y : msg->goal_post_cam_y) / 1000.0;
    const double z = (is_pick_cycle ? msg->ball_cam_z : msg->goal_post_cam_z) / 1000.0;
    fsm_->update_vision(x, y, z);
  }

  // 이벤트만 기록함(내부적으로 자체 락 보호) - 실제 처리(onMotionEndImpl)는
  // on_solve_tick()이 take_pending_motion_end()로 소비할 때 일어남.
  void on_motion_end(const MotionOperator::SharedPtr msg) {
    fsm_->notify_motion_end(msg->motion_end);
  }

  // fsm_ 로직(on_motion_end/on_motor_feedback/solve_tick)을 태우는 유일한 곳 -
  // 전용 콜백 그룹(solve_callback_group_)의 스레드 하나에서만 돌아서 이 셋끼리는
  // 서로 레이스가 안 남. 대기 중인 motion_end가 있으면 먼저 소비해서 항상 적용.
  // 그다음 on_motor_feedback()이 이번 tick에 실제로 보낼 모터 명령이 있으면
  // (그립 정지 판정/그립 위글 등) 그것만 적용하고 끝 - solve_tick(IK)은 이번
  // tick엔 안 돌림. on_motor_feedback이 딱히 할 게 없을 때만 solve_tick을 돌려서
  // 그 결과를 적용함(한 tick에 모터 명령이 최대 1번만 나가게).
  void on_solve_tick() {
    // COMPLETE_PLACE 완료 후 deactivate 콜백이 실행되기 전까지
    // 다음 solve tick이 들어와도 FSM/퍼블리셔를 다시 건드리지 않는다.
    if (deactivate_pending_) {
      return;
    }

    // 아래 분기들이 조기 return하더라도 매 tick 빠짐없이 갱신하고 싶어서 맨 앞에서 호출.
    publish_debug_markers();

    if (const std::optional<bool> pending_motion_end = fsm_->take_pending_motion_end();
      pending_motion_end.has_value())
    {
      apply_action(fsm_->on_motion_end(*pending_motion_end));
    }

    const FsmAction motor_feedback_action = fsm_->on_motor_feedback();
    if (motor_feedback_action.publish_motor_command) {
      apply_action(motor_feedback_action);
      return;
    }
    apply_action(fsm_->solve_tick());
  }

  // RViz 디버그 시각화 전용 - target_pos_는 큰 구(SPHERE) 하나, 활성 궤적(PICK.PICK의
  // pick_approach_trajectory_ 또는 PLACE.VIRTUAL_PLACE의 place_trajectory_) 샘플들은
  // 작은 점들(POINTS, 한 Marker에 다 담음)로 publish함. 판단 없이 fsm_ 값을 그대로
  // 옮기기만 하는 apply_action()과 동일한 성격이라 여기 같이 둠.
  void publish_debug_markers() {
    const auto stamp = get_clock()->now();
    const Eigen::Vector3d target = fsm_->target_pos();

    Marker target_marker;
    target_marker.header.frame_id = "base_link";
    target_marker.header.stamp = stamp;
    target_marker.ns = "manipulation_target";
    target_marker.id = 0;
    target_marker.type = Marker::SPHERE;
    target_marker.action = Marker::ADD;
    target_marker.pose.position.x = target.x();
    target_marker.pose.position.y = target.y();
    target_marker.pose.position.z = target.z();
    target_marker.pose.orientation.w = 1.0;
    target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.08;
    target_marker.color.r = 1.0f;
    target_marker.color.a = 1.0f;
    target_marker_pub_->publish(target_marker);

    Marker trajectory_marker;
    trajectory_marker.header.frame_id = "base_link";
    trajectory_marker.header.stamp = stamp;
    trajectory_marker.ns = "manipulation_trajectory";
    trajectory_marker.id = 0;
    trajectory_marker.type = Marker::POINTS;
    // 활성 궤적이 없는 상태(빈 벡터)면 DELETE로 이전에 찍어둔 점들을 지움 - ADD에
    // points가 비어있으면 그냥 아무것도 안 그려질 뿐 이전 마커가 안 지워짐.
    const std::vector<Eigen::Vector3d> samples = fsm_->sample_active_trajectory(1000);
    trajectory_marker.action = samples.empty() ? Marker::DELETE : Marker::ADD;
    trajectory_marker.pose.orientation.w = 1.0;
    trajectory_marker.scale.x = trajectory_marker.scale.y = 0.01;
    trajectory_marker.color.b = 1.0f;
    trajectory_marker.color.a = 1.0f;
    trajectory_marker.points.reserve(samples.size());
    for (const Eigen::Vector3d & sample : samples) {
      geometry_msgs::msg::Point point;
      point.x = sample.x();
      point.y = sample.y();
      point.z = sample.z();
      trajectory_marker.points.push_back(point);
    }
    trajectory_marker_pub_->publish(trajectory_marker);
  }

  // fsm_가 반환한 액션을 기계적으로 ROS 액션으로 옮김 - 판단 없음. action은 이미
  // 스냅샷된 값이라 fsm_에 다시 접근하지 않음.
  void apply_action(const FsmAction & action) {
    if (action.publish_motor_command) {
      const auto stamp = get_clock()->now();

      DynamixelControlMsgs control_msg;
      control_msg.motor_control.reserve(kMotorIdOrder.size() - 1);
      for (const auto & motor_id : kMotorIdOrder) {
        if (motor_id == kGripMotorId) {
          continue;
        }
        DynamixelMsgs motor_msg;
        motor_msg.header.stamp = stamp;
        auto it = action.motor_goal_positions.find(motor_id);
        motor_msg.goal_position =
          static_cast<float>(it != action.motor_goal_positions.end() ? it->second : 0.0);
        motor_msg.profile_acceleration = 0.0f;
        motor_msg.profile_velocity = static_cast<float>(action.profile_velocity);
        control_msg.motor_control.push_back(motor_msg);
      }
      control_pub_->publish(control_msg);

      DynamixelMsgs gripper_msg;
      gripper_msg.header.stamp = stamp;
      auto grip_it = action.motor_goal_positions.find(kGripMotorId);
      gripper_msg.goal_position =
        static_cast<float>(grip_it != action.motor_goal_positions.end() ? grip_it->second : 0.0);
      gripper_msg.profile_acceleration = 0.0f;
      gripper_msg.profile_velocity = static_cast<float>(action.profile_velocity);
      gripper_pub_->publish(gripper_msg);
    }

    if (action.play_motion_number.has_value()) {
      MotionOperator msg;
      msg.motion_num = *action.play_motion_number;
      motion_operator_pub_->publish(msg);
      RCLCPP_INFO(
        get_logger(), "motion_operator -> motion_num=%d (%s)",
        *action.play_motion_number, action.state_string.c_str());
    }

    if (action.deactivate_and_notify) {
      Bool activate_end;
      activate_end.data = true;
      activate_end_pub_->publish(activate_end);

      // lifecycle deactivate()를 solve_timer_ 콜백 안에서 동기 호출하면
      // executor/lifecycle이 현재 콜백의 타이머와 pub/sub를 정리하며
      // use-after-free가 날 수 있다. 같은 MutuallyExclusive 콜백 그룹에
      // one-shot 타이머를 예약해 현재 solve 콜백이 끝난 후 전환한다.
      if (!deactivate_pending_) {
        deactivate_pending_ = true;
        deferred_deactivate_timer_ = create_wall_timer(
          std::chrono::milliseconds(1),
          [this]() {
            deferred_deactivate_timer_->cancel();
            this->deactivate();
          },
          solve_callback_group_);
      }
    }

    if (action.reset_pan_tilt_level) {
      // realsense_pan_tilt_node에게 place용 tilt 모드(5)를 요청 - 그 노드가 모드를
      // 실제 각도로 변환해서 /RealsensePanTilt에 다시 publish함.
      Master2VisionMsg pan_tilt_msg;
      pan_tilt_msg.tilt = kPlaceTiltMode;
      pan_tilt_pub_->publish(pan_tilt_msg);
    }
  }

  rclcpp::Subscription<Bool>::SharedPtr activate_cmd_sub_;
  rclcpp::Publisher<Bool>::SharedPtr activate_end_pub_;
  rclcpp::Publisher<DynamixelControlMsgs>::SharedPtr control_pub_;
  rclcpp::Publisher<DynamixelMsgs>::SharedPtr gripper_pub_;
  rclcpp::Publisher<Master2VisionMsg>::SharedPtr pan_tilt_pub_;
  rclcpp::Publisher<Marker>::SharedPtr target_marker_pub_;
  rclcpp::Publisher<Marker>::SharedPtr trajectory_marker_pub_;
  rclcpp::Subscription<CurrentMotorStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<Vision2ManiMsg>::SharedPtr vision_sub_;
  rclcpp::Publisher<MotionOperator>::SharedPtr motion_operator_pub_;
  rclcpp::Subscription<MotionOperator>::SharedPtr motion_end_sub_;
  rclcpp::CallbackGroup::SharedPtr solve_callback_group_;
  rclcpp::TimerBase::SharedPtr solve_timer_;
  rclcpp::TimerBase::SharedPtr deferred_deactivate_timer_;
  bool deactivate_pending_ = false;

  std::unique_ptr<ManipulationFSM> fsm_;

  std::string control_topic_;
  std::string gripper_topic_;
  std::string status_topic_;
  std::string pan_tilt_topic_;
  double step_period_sec_ = 0.02;
  // on_activate()마다 1씩 증가 - on_vision2mani()가 이번이 몇 번째 활성화인지로
  // ball_cam/goal_post_cam 중 어느 쪽을 넘길지 판단하는 데 씀.
  int activate_count_ = 0;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ManipulationLifecycleNode>();

  // IK 솔브 타이머(전용 콜백 그룹)가 motor_status 등 메시지 콜백과 다른 스레드에서
  // 돌게 스레드 2개짜리 MultiThreadedExecutor로 스핀함 - 원래 이 노드와
  // ik_solve_node.py를 별도 프로세스로 뒀던 이유(IK 연산이 느려도 모터 I/O가
  // 막히지 않게)를 스레드 수준에서 재현.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node->get_node_base_interface());
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
