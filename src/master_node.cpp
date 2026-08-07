// 실기 master(task_planner) 노드를 흉내내는 작은 테스트용 노드. debug_node.cpp처럼
// /vision2mani(Vision2ManiMsg, 매니퓰레이션 전용 비전 출력 - 공/골대 둘 다 x/y/z를
// 줌)를 직접 구독하되, main_node.cpp에 바로 물릴 수 있도록 실제 master가 하는
// 역할(비전 -> /master2mani 릴레이 + activate_cmd 발행)만 최소한으로 재현함.
// main_node.cpp는 그대로 두고 이 노드만 새로 띄우면 실제 master/task_planner
// 없이도 전체 파이프라인을 테스트할 수 있음.
//
//  - 's'/'d' 키로 지금부터 추적할 대상(공/골대)을 고르고 activate_cmd(true)를
//    한 번 보냄(main_node를 SIT -> PICK_READY로 깨움). 이후로는 /vision2mani를
//    받는 즉시(매 메시지마다) 그 대상의 좌표를 m로 변환해 /master2mani(Point32,
//    main_node.cpp가 구독하는 것과 동일한 토픽/타입)로 계속 릴레이함 - 's': 공
//    (ball_cam_x/y/z), 'd': 골대(goal_post_cam_x/y/z). 아직 아무 키도 안 눌렀으면
//    /vision2mani가 와도 아무것도 안 보냄(target_ == kNone).
//  - main_node.cpp가 픽/플레이스 사이클 끝나고 스스로 deactivate하면서
//    activate_cmd에 false를 보내는데, 그걸 받으면 터미널에 로그로 알려줌
//    (activate_cmd_pub_/sub_는 main_node.cpp와 마찬가지로 양방향 - 여기서도
//    같은 토픽을 구독해야 그 알림을 받을 수 있음).
//  - 'd' 키를 누른 그 순간에만 딱 한 번(반복 아님), 그리퍼 모터 토픽
//    (gripper_topic, main_node.cpp/debug_ik_node.cpp와 동일하게 기본 /gripper_control)
//    으로 gripper.closed_deg에 해당하는 위치를 DynamixelMsgs로 명령함 - 골대 앞에서
//    바로 손을 닫아 공을 놓기 전 테스트용.

#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "dynamixel_hardware_msgs/msg/dynamixel_msgs.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "irc_humanoid_interfaces/msg/vision2_mani_msg.hpp"
#include "std_msgs/msg/bool.hpp"

using dynamixel_hardware_msgs::msg::DynamixelMsgs;
using geometry_msgs::msg::Point32;
using irc_humanoid_interfaces::msg::Vision2ManiMsg;
using std_msgs::msg::Bool;

namespace {

// debug_ik_node.cpp의 getKey와 동일: raw 모드로 잠깐 들어가서 timeout 동안 문자
// 하나를 기다리고, 읽든 못 읽든 매번 원래(cooked) 모드로 복원함.
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

}  // namespace

class MasterNode : public rclcpp::Node {
 public:
  enum class Target { kNone, kBall, kGoalPost };

  MasterNode() : rclcpp::Node("master_node") {
    // main_node.cpp의 mani_sub_와 동일한 QoS(depth=4)로 맞춤.
    mani_pub_ = create_publisher<Point32>("/master2mani", rclcpp::QoS(4));
    vision_sub_ = create_subscription<Vision2ManiMsg>(
        "/vision2mani", rclcpp::QoS(10),
        [this](const Vision2ManiMsg::SharedPtr msg) { onVision(msg); });

    // main_node.cpp의 activate_cmd_sub_/activate_cmd_pub_와 동일한 QoS(유실 허용
    // 안 됨 -> RELIABLE). 이 노드도 같은 토픽을 펴블리시(true, 's'/'d' 입력 시)/구독
    // (false, main_node가 deactivate 통지) 양쪽 다 함.
    auto activate_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    activate_cmd_pub_ = create_publisher<Bool>("activate_cmd", activate_qos);
    activate_cmd_sub_ = create_subscription<Bool>(
        "activate_cmd", activate_qos,
        [this](const Bool::SharedPtr msg) { onActivateCmd(msg); });

    // main_node.cpp/debug_ik_node.cpp와 동일한 토픽/QoS/변환(deg -> rad).
    const std::string gripper_topic =
        declare_parameter<std::string>("gripper_topic", "/gripper_control");
    const double gripper_closed_deg = declare_parameter<double>("gripper.closed_deg", -36.7033);
    gripper_closed_rad_ = gripper_closed_deg * M_PI / 180.0;
    gripper_pub_ = create_publisher<DynamixelMsgs>(gripper_topic, rclcpp::QoS(4));
  }

  // 's': 이후 /vision2mani가 올 때마다 공 좌표를 추적/릴레이하도록 선택 + activate.
  void selectBallAndActivate() {
    target_ = Target::kBall;
    sendActivate();
    RCLCPP_INFO(get_logger(), "추적 대상: 공 (앞으로 /vision2mani 받을 때마다 릴레이).");
  }

  // 'd': 이후 /vision2mani가 올 때마다 골대 좌표를 추적/릴레이하도록 선택 + activate +
  // 그리퍼를 closed_deg 위치로 딱 한 번 명령(반복 아님).
  void selectGoalPostAndActivate() {
    target_ = Target::kGoalPost;
    sendActivate();
    RCLCPP_INFO(get_logger(), "추적 대상: 골대 (앞으로 /vision2mani 받을 때마다 릴레이).");

    DynamixelMsgs gripper_msg;
    gripper_msg.header.stamp = get_clock()->now();
    gripper_msg.goal_position = static_cast<float>(gripper_closed_rad_);
    gripper_msg.profile_acceleration = 0.0f;
    gripper_msg.profile_velocity = 1.0f;
    gripper_pub_->publish(gripper_msg);
    RCLCPP_INFO(get_logger(), "그리퍼 closed_deg 위치(%.4frad) 명령.", gripper_closed_rad_);
  }

 private:
  void sendActivate() {
    Bool activate_msg;
    activate_msg.data = true;
    activate_cmd_pub_->publish(activate_msg);
    RCLCPP_INFO(get_logger(), "activate_cmd(true) 전송.");
  }

  void onVision(const Vision2ManiMsg::SharedPtr msg) {
    // target_이 아직 안 정해졌으면(키를 한 번도 안 눌렀으면) 아무것도 안 보냄.
    if (target_ == Target::kNone) {
      return;
    }
    Point32 out;
    if (target_ == Target::kBall) {
      out.x = static_cast<float>(msg->ball_cam_x / 1000.0);
      out.y = static_cast<float>(msg->ball_cam_y / 1000.0);
      out.z = static_cast<float>(msg->ball_cam_z / 1000.0);
    } else {
      out.x = static_cast<float>(msg->goal_post_cam_x / 1000.0);
      out.y = static_cast<float>(msg->goal_post_cam_y / 1000.0);
      out.z = static_cast<float>(msg->goal_post_cam_z / 1000.0);
    }
    mani_pub_->publish(out);
  }

  void onActivateCmd(const Bool::SharedPtr msg) {
    // false는 main_node가 픽/플레이스 사이클을 끝내고 스스로 deactivate했다는
    // 통지 - 이 노드가 방금 보낸 true는 그대로 무시(로그 대상 아님).
    if (!msg->data) {
      RCLCPP_INFO(get_logger(), "main_node로부터 deactivate 신호(activate_cmd=false) 받음.");
    }
  }

  rclcpp::Publisher<Point32>::SharedPtr mani_pub_;
  rclcpp::Subscription<Vision2ManiMsg>::SharedPtr vision_sub_;
  rclcpp::Publisher<Bool>::SharedPtr activate_cmd_pub_;
  rclcpp::Subscription<Bool>::SharedPtr activate_cmd_sub_;
  rclcpp::Publisher<DynamixelMsgs>::SharedPtr gripper_pub_;
  double gripper_closed_rad_ = 0.0;
  Target target_ = Target::kNone;
};

namespace {

void keyboardLoop(const std::shared_ptr<MasterNode>& node) {
  termios original;
  tcgetattr(STDIN_FILENO, &original);
  std::cout << "s: 공 추적 선택 + activate_cmd(true)  d: 골대 추적 선택 + "
               "activate_cmd(true)  (Ctrl+C: 종료)\n";
  try {
    while (rclcpp::ok()) {
      const char key = getKey(original, 0.1);
      if (key == 's') {
        node->selectBallAndActivate();
      } else if (key == 'd') {
        node->selectGoalPostAndActivate();
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
  auto node = std::make_shared<MasterNode>();

  std::thread spin_thread([node]() { rclcpp::spin(node); });

  keyboardLoop(node);

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
