// 실기 master(task_planner) 노드를 흉내내는 작은 테스트용 노드. debug_node.cpp처럼
// /vision2mani(Vision2ManiMsg, 매니퓰레이션 전용 비전 출력 - 공/골대 둘 다 x/y/z를
// 줌)를 직접 구독하되, main_node.cpp에 바로 물릴 수 있도록 실제 master가 하는
// 역할(비전 -> /master2mani 릴레이 + activate_cmd 발행)만 최소한으로 재현함.
// main_node.cpp는 그대로 두고 이 노드만 새로 띄우면 실제 master/task_planner
// 없이도 전체 파이프라인을 테스트할 수 있음.
//
//  - /vision2mani를 받으면 지금까지 받은 deactivate(activate_cmd=false) 횟수의 홀/짝에
//    따라 공(ball_cam_x/y/z) 또는 미니 농구대 백보드(goal_post_cam_x/y/z) 좌표를
//    m로 변환해 /master2mani(Point32, main_node.cpp가 구독하는 것과 동일한
//    토픽/타입)로 보냄 - PICK 사이클(홀수번째, deactivate 0회 포함)은 공을,
//    PLACE 사이클(짝수번째)은 골대를 목표로 삼는다고 가정함.
//  - 터미널에서 's' 키를 누르면 그때 activate_cmd(true)를 한 번 보내서 main_node를
//    SIT -> PICK_READY로 깨움(자동 타이머 아님 - 원할 때 직접 트리거).
//  - main_node.cpp가 픽 시퀀스(DONE) 끝나고 스스로 deactivate하면서 activate_cmd에
//    false를 보내는데, 그걸 받으면 터미널에 로그로 알려주고 다음 사이클의 비전
//    타겟(공/골대)을 뒤집는 카운터를 증가시킴(activate_cmd_pub_/sub_는
//    main_node.cpp와 마찬가지로 양방향 - 여기서도 같은 토픽을 구독해야 그 알림을
//    받을 수 있음).

#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

#include <iostream>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "geometry_msgs/msg/point32.hpp"
#include "irc_humanoid_interfaces/msg/vision2_mani_msg.hpp"
#include "std_msgs/msg/bool.hpp"

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
  MasterNode() : rclcpp::Node("master_node") {
    // main_node.cpp의 mani_sub_와 동일한 QoS(depth=4)로 맞춤.
    mani_pub_ = create_publisher<Point32>("/master2mani", rclcpp::QoS(4));
    vision_sub_ = create_subscription<Vision2ManiMsg>(
        "/vision2mani", rclcpp::QoS(10),
        [this](const Vision2ManiMsg::SharedPtr msg) { onVision(msg); });

    // main_node.cpp의 activate_cmd_sub_/activate_cmd_pub_와 동일한 QoS(유실 허용
    // 안 됨 -> RELIABLE). 이 노드도 같은 토픽을 펴블리시(true, 's' 입력 시)/구독
    // (false, main_node가 deactivate 통지) 양쪽 다 함.
    auto activate_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    activate_cmd_pub_ = create_publisher<Bool>("activate_cmd", activate_qos);
    activate_cmd_sub_ = create_subscription<Bool>(
        "activate_cmd", activate_qos,
        [this](const Bool::SharedPtr msg) { onActivateCmd(msg); });
  }

  void sendActivate() {
    Bool msg;
    msg.data = true;
    activate_cmd_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "activate_cmd(true) 전송.");
  }

 private:
  void onVision(const Vision2ManiMsg::SharedPtr msg) {
    // deactivate_count_가 짝수(0 포함)면 홀수번째 사이클(PICK) -> 공, 홀수면
    // 짝수번째 사이클(PLACE) -> 골대. 둘 다 mm 단위로 옴 - main_node/IK는 m
    // 단위라 여기서 변환.
    Point32 out;
    if (deactivate_count_ % 2 == 0) {
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
    // 통지 - 이 노드가 방금 보낸 true는 그대로 무시(로그 대상 아님). 다음
    // 사이클의 비전 타겟(공/골대)을 뒤집기 위해 카운트만 늘림.
    if (!msg->data) {
      ++deactivate_count_;
      RCLCPP_INFO(get_logger(),
                  "main_node로부터 deactivate 신호(activate_cmd=false) 받음(%d번째) -> "
                  "다음 사이클 비전 타겟: %s",
                  deactivate_count_, (deactivate_count_ % 2 == 0) ? "공" : "골대");
    }
  }

  rclcpp::Publisher<Point32>::SharedPtr mani_pub_;
  rclcpp::Subscription<Vision2ManiMsg>::SharedPtr vision_sub_;
  rclcpp::Publisher<Bool>::SharedPtr activate_cmd_pub_;
  rclcpp::Subscription<Bool>::SharedPtr activate_cmd_sub_;
  int deactivate_count_ = 0;
};

namespace {

void keyboardLoop(const std::shared_ptr<MasterNode>& node) {
  termios original;
  tcgetattr(STDIN_FILENO, &original);
  std::cout << "s: activate_cmd(true) 전송  (Ctrl+C: 종료)\n";
  try {
    while (rclcpp::ok()) {
      const char key = getKey(original, 0.1);
      if (key == 's') {
        node->sendActivate();
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
