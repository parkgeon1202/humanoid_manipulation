// main_node.cpp와 완전히 같은 로직(FSM+IK+모터 I/O, 전부 fsm.cpp에 있음)을 쓰는
// 디버그용 노드. main_node.cpp와의 차이는 딱 두 가지뿐:
//  1) activate_cmd(task_planner)를 기다리지 않고 프로세스가 뜨자마자 바로 activate.
//  2) 비전 정보를 /master2mani(Point32, master 경유)가 아니라 /vision
//     (Vision2MasterMsg, 비전 노드가 직접 보내는 원본)에서 바로 받음.
// -> master/task_planner 없이 비전->FSM->모터 파이프라인 전체를 단독으로 테스트하는 용도.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "dynamixel_hardware_msgs/msg/current_motor_status.hpp"
#include "dynamixel_hardware_msgs/msg/dynamixel_control_msgs.hpp"
#include "fsm.hpp"
#include "irc_humanoid_interfaces/msg/motion_operator.hpp"
#include "irc_humanoid_interfaces/msg/vision2_master_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/bool.hpp"

using CallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using dynamixel_hardware_msgs::msg::CurrentMotorStatus;
using dynamixel_hardware_msgs::msg::DynamixelControlMsgs;
using irc_humanoid_interfaces::msg::MotionOperator;
using irc_humanoid_interfaces::msg::Vision2MasterMsg;
using rclcpp_lifecycle::State;
using std_msgs::msg::Bool;

namespace
{
// tilt(23번)는 이 시스템에서 아예 제어 안 해서 목록에서 뺌(fsm.cpp와 동일해야 함).
const std::vector<std::string> kMotorIdOrder = {
  "0", "1", "2", "3", "4", "5", "6",
  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
  "22"};

std::string resolveUrdfPath()
{
  return ament_index_cpp::get_package_share_directory("humanoid_manipulation") +
         "/model/irc_man.urdf";
}
}  // namespace

class DebugManipulationLifecycleNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  DebugManipulationLifecycleNode() : rclcpp_lifecycle::LifecycleNode("debug_node") {
    // main_node.cpp와 달리 activate_cmd를 기다리지 않고 configure/activate를
    // 프로세스가 뜨자마자 바로 진행함(단독 테스트용).
    this->configure();
    this->activate();
  }

  CallbackReturn on_configure(const State &) override {
    declare_and_get_params();
    create_activate_cmd_sub();

    RCLCPP_INFO(get_logger(), "Configured. Waiting for activate_cmd.");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const State &) override {
    auto motor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    control_pub_ = create_publisher<DynamixelControlMsgs>(control_topic_, motor_qos);
    status_sub_ = create_subscription<CurrentMotorStatus>(
      status_topic_, motor_qos,
      [this](const CurrentMotorStatus::SharedPtr msg) {on_motor_status(msg);});

    // 비전 노드가 직접 보내는 /vision(Vision2MasterMsg)에서 공 위치를 바로 받음
    // (main_node.cpp는 master 경유인 /master2mani를 받지만, 여기는 단독 테스트용
    // 이라 원본 비전 토픽을 직접 구독함).
    vision_sub_ = create_subscription<Vision2MasterMsg>(
      "/vision", 10,
      [this](const Vision2MasterMsg::SharedPtr msg) {on_vision(msg);});

    auto motion_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    motion_operator_pub_ = create_publisher<MotionOperator>("motion_operator", motion_qos);
    motion_end_sub_ = create_subscription<MotionOperator>(
      "motion_end", motion_qos,
      [this](const MotionOperator::SharedPtr msg) {on_motion_end(msg);});

    solve_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    solve_timer_ = create_wall_timer(
      std::chrono::duration<double>(step_period_sec_),
      [this]() {on_solve_tick();},
      solve_callback_group_);

    {
      std::lock_guard<std::mutex> lock(fsm_mutex_);
      apply_action(fsm_->on_activated());
    }

    RCLCPP_INFO(
      get_logger(), "Activated. Listening on /vision, %s / %s.",
      control_topic_.c_str(), status_topic_.c_str());
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const State &) override {
    solve_timer_.reset();
    control_pub_.reset();
    status_sub_.reset();
    vision_sub_.reset();
    motion_operator_pub_.reset();
    motion_end_sub_.reset();

    RCLCPP_INFO(get_logger(), "Deactivated. Waiting for activate_cmd.");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_cleanup(const State &) override {
    activate_cmd_sub_.reset();
    activate_cmd_pub_.reset();
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
    status_topic_ = declare_parameter<std::string>("status_topic", "/motor_status");
    step_period_sec_ = declare_parameter<double>("ik.step_period_sec", 0.02);

    IkTuning tuning;
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
    const double gripper_closed_deg = declare_parameter<double>("gripper.closed_deg", 0.0);
    const double gripper_step_deg = declare_parameter<double>("gripper.step_deg", 0.6);
    tuning.gripper_open_rad = gripper_open_deg * M_PI / 180.0;
    tuning.gripper_closed_rad = gripper_closed_deg * M_PI / 180.0;
    tuning.gripper_step_rad = gripper_step_deg * M_PI / 180.0;

    // [torso_yaw, left_shoulder_pitch, left_shoulder_roll, left_elbow_pitch] -
    // left_shoulder_roll 기본값은 ik.left_shoulder_roll_target_rad와 동일하게 1.0.
    const std::vector<double> home_q = declare_parameter<std::vector<double>>(
      "pick_ready_home_q", std::vector<double>{0.0, 0.0, 1.0, 0.0});
    const std::vector<double> joint_weights_4 = declare_parameter<std::vector<double>>(
      "ik.joint_weights", std::vector<double>{1.0, 1.0, 1.0, 1.0});
    tuning.joint_weight_scale = declare_parameter<double>("ik.joint_weight_scale", 0.0);
    tuning.secondary_gain = declare_parameter<double>("ik.secondary_gain", 0.0);
    tuning.collision_avoidance_threshold_m =
      declare_parameter<double>("ik.collision_avoidance_threshold_m", 0.03);
    tuning.collision_avoidance_k_pull =
      declare_parameter<double>("ik.collision_avoidance_k_pull", 1.0);
    tuning.stuck_streak_ticks = declare_parameter<int>("ik.stuck_streak_ticks", 30);
    tuning.torso_kick_offsets = declare_parameter<std::vector<double>>(
      "ik.torso_kick_offsets", std::vector<double>{0.5, 1.0, 1.5, -0.5, -1.0, -1.5});

    tuning.torso_target_rad = declare_parameter<double>("ik.torso_target_rad", -1.5);
    tuning.left_shoulder_roll_target_rad =
      declare_parameter<double>("ik.left_shoulder_roll_target_rad", 1.0);
    tuning.settle_velocity_threshold =
      declare_parameter<double>("ik.torso_settle_velocity_threshold", 0.3);
    tuning.settle_position_tolerance_rad =
      declare_parameter<double>("ik.torso_settle_position_tolerance_rad", 0.05);

    tuning.grip_moving_velocity_threshold =
      declare_parameter<double>("grip.moving_velocity_threshold", 0.3);
    tuning.grip_near_zero_position_threshold =
      declare_parameter<double>("grip.near_zero_position_threshold", 0.2);
    tuning.grip_wiggle_kick_position_rad =
      declare_parameter<double>("grip.wiggle_kick_position_rad", 1.0);

    const int nq = 8;
    tuning.home_q_full = Eigen::VectorXd::Zero(nq);
    tuning.joint_weights = Eigen::VectorXd::Ones(nq);
    if (home_q.size() >= 4) {
      tuning.home_q_full[ik::kTorsoJointIndex] = home_q[0];
      for (int i = 0; i < 3; ++i) tuning.home_q_full[ik::kLeftArmIndices[i]] = home_q[i + 1];
    }
    if (joint_weights_4.size() >= 4) {
      tuning.joint_weights[ik::kTorsoJointIndex] = joint_weights_4[0];
      for (int i = 0; i < 3; ++i) tuning.joint_weights[ik::kLeftArmIndices[i]] = joint_weights_4[i + 1];
    }

    fsm_->configure_ik(tuning);
  }

  void create_activate_cmd_sub() {
    auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    activate_cmd_sub_ = create_subscription<Bool>(
      "activate_cmd", cmd_qos,
      [this](const Bool::SharedPtr msg) {on_activate_cmd(msg);});
    activate_cmd_pub_ = create_publisher<Bool>("activate_cmd", cmd_qos);
  }

  void on_activate_cmd(const Bool::SharedPtr msg) {
    if (msg->data) {
      this->activate();
    } else {
      this->deactivate();
    }
  }

  void on_motor_status(const CurrentMotorStatus::SharedPtr msg) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "motor_status received: %zu motor(s)", msg->position.size());

    std::lock_guard<std::mutex> lock(fsm_mutex_);
    const size_t n = std::min(kMotorIdOrder.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i) {
      fsm_->set_motor_position(kMotorIdOrder[i], msg->position[i]);
    }
    const size_t nv = std::min(kMotorIdOrder.size(), msg->velocity.size());
    for (size_t i = 0; i < nv; ++i) {
      fsm_->set_motor_velocity(kMotorIdOrder[i], msg->velocity[i]);
    }
    apply_action(fsm_->on_motor_feedback());
  }

  void on_vision(const Vision2MasterMsg::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(fsm_mutex_);
    fsm_->update_vision(msg->ball_cam_x, msg->ball_cam_y, msg->ball_cam_z);
  }

  void on_motion_end(const MotionOperator::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(fsm_mutex_);
    apply_action(fsm_->on_motion_end(msg->motion_end));
  }

  void on_solve_tick() {
    std::lock_guard<std::mutex> lock(fsm_mutex_);
    apply_action(fsm_->solve_tick());
  }

  void apply_action(const FsmAction & action) {
    if (action.publish_motor_command) {
      DynamixelControlMsgs control_msg;
      const auto stamp = get_clock()->now();
      control_msg.motor_control.resize(kMotorIdOrder.size());
      for (size_t i = 0; i < kMotorIdOrder.size(); ++i) {
        control_msg.motor_control[i].header.stamp = stamp;
        auto it = action.motor_goal_positions.find(kMotorIdOrder[i]);
        control_msg.motor_control[i].goal_position =
          static_cast<float>(it != action.motor_goal_positions.end() ? it->second : 0.0);
        control_msg.motor_control[i].profile_acceleration = 0.0f;
        control_msg.motor_control[i].profile_velocity =
          static_cast<float>(action.profile_velocity);
      }
      control_pub_->publish(control_msg);
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
      Bool activate_ack;
      activate_ack.data = false;
      activate_cmd_pub_->publish(activate_ack);
      this->deactivate();
    }
  }

  rclcpp::Subscription<Bool>::SharedPtr activate_cmd_sub_;
  rclcpp::Publisher<Bool>::SharedPtr activate_cmd_pub_;
  rclcpp::Publisher<DynamixelControlMsgs>::SharedPtr control_pub_;
  rclcpp::Subscription<CurrentMotorStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<Vision2MasterMsg>::SharedPtr vision_sub_;
  rclcpp::Publisher<MotionOperator>::SharedPtr motion_operator_pub_;
  rclcpp::Subscription<MotionOperator>::SharedPtr motion_end_sub_;
  rclcpp::CallbackGroup::SharedPtr solve_callback_group_;
  rclcpp::TimerBase::SharedPtr solve_timer_;

  std::unique_ptr<ManipulationFSM> fsm_;
  std::mutex fsm_mutex_;

  std::string control_topic_;
  std::string status_topic_;
  double step_period_sec_ = 0.02;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DebugManipulationLifecycleNode>();

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node->get_node_base_interface());
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
