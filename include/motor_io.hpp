// kid_rl(하드웨어 인터페이스)와 주고받는 모터 메시지의 id 처리 - main_node.cpp,
// debug_ik_node.cpp, master_node.cpp가 같은 규칙을 쓰도록 한 곳에 모음.
//
// 상태(/motor_status, DynamixelStateMsgs) 받기:
//  - 배열 순서는 안 씀. id -> {position, velocity} 표로 바꿔서 id로 찾음.
//  - header.stamp가 0이거나 너무 오래됐거나(미래 포함), 중복 id, 유한하지 않은 값,
//    이 노드가 쓰는 id(required_ids) 중 하나라도 빠지면 메시지 전체를 버림 -
//    일부만 갱신하면 관절끼리 다른 시점 값이 섞임.
//  - 이 노드가 안 쓰는 id는 그냥 무시(거부 안 함).
// 명령(/dynamixel_control, /gripper_control) 보내기:
//  - 원소마다 id를 채움. kid_rl은 모르는/중복 id, 유한하지 않은 값, 음수 profile,
//    빈 배열이 하나라도 있으면 메시지 전체를 버리므로 보내기 전에 같은 검사를 함.

#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "dynamixel_interfaces/msg/dynamixel_control_msgs.hpp"
#include "dynamixel_interfaces/msg/dynamixel_msgs.hpp"
#include "dynamixel_interfaces/msg/dynamixel_state_msgs.hpp"
#include "rclcpp/time.hpp"

namespace motor_io
{

// Dynamixel ID 범위(253 이상은 예약/브로드캐스트).
constexpr int kMaxMotorId = 252;
// 같은 PC의 kid_rl이 stamp를 찍으므로 이보다 미래면 시계/시간원 설정이 어긋난 것.
constexpr double kFutureToleranceSec = 0.05;

struct MotorState
{
  double position = 0.0;
  double velocity = 0.0;
};

// 문자열 모터 id("0", "22" 등, fsm.cpp/kMotorIdOrder 규칙)를 숫자로. 범위 밖이거나
// 숫자가 아니면 nullopt.
inline std::optional<uint8_t> parse_motor_id(const std::string & motor_id)
{
  if (motor_id.empty() || motor_id.size() > 3) {
    return std::nullopt;
  }
  int id = 0;
  for (const char c : motor_id) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    id = id * 10 + (c - '0');
  }
  if (id > kMaxMotorId) {
    return std::nullopt;
  }
  return static_cast<uint8_t>(id);
}

// /motor_status 한 개를 검사해서 id -> 상태 표로 바꿈. 실패하면 false + reason.
// required_ids는 이 노드가 실제로 계산에 쓰는 모터만.
inline bool decode_status(
  const dynamixel_interfaces::msg::DynamixelStateMsgs & msg, const rclcpp::Time & now,
  double timeout_sec, const std::vector<uint8_t> & required_ids,
  std::map<uint8_t, MotorState> & out, std::string & reason)
{
  out.clear();
  // stamp는 받는 쪽 시계 종류로 해석해야 now와 뺄셈이 됨(종류가 다르면 예외).
  const rclcpp::Time stamp(msg.header.stamp, now.get_clock_type());
  if (stamp.nanoseconds() == 0) {
    reason = "header.stamp is zero";
    return false;
  }
  const double age_sec = (now - stamp).seconds();
  if (age_sec > timeout_sec) {
    reason = "stale: " + std::to_string(age_sec) + " s old";
    return false;
  }
  if (age_sec < -kFutureToleranceSec) {
    reason = "header.stamp is " + std::to_string(-age_sec) + " s in the future";
    return false;
  }

  for (const auto & state : msg.motor_state) {
    if (!std::isfinite(state.position) || !std::isfinite(state.velocity)) {
      reason = "id " + std::to_string(state.id) + " has a non-finite value";
      return false;
    }
    if (!out.emplace(state.id, MotorState{state.position, state.velocity}).second) {
      reason = "duplicated id " + std::to_string(state.id);
      return false;
    }
  }

  std::string missing;
  for (const uint8_t id : required_ids) {
    if (out.count(id) == 0) {
      missing += " " + std::to_string(id);
    }
  }
  if (!missing.empty()) {
    reason = "missing id(s):" + missing;
    return false;
  }
  return true;
}

// 명령 원소 하나. 값이 유한하지 않거나 profile이 음수면 false + reason.
inline bool make_command(
  uint8_t id, double goal_position, double profile_velocity, const rclcpp::Time & stamp,
  dynamixel_interfaces::msg::DynamixelMsgs & out, std::string & reason)
{
  if (!std::isfinite(goal_position) || !std::isfinite(profile_velocity) ||
    profile_velocity < 0.0)
  {
    reason = "id " + std::to_string(id) + ": goal " + std::to_string(goal_position) +
      " / profile_velocity " + std::to_string(profile_velocity) + " is invalid";
    return false;
  }
  out.header.stamp = stamp;
  out.id = id;
  out.goal_position = static_cast<float>(goal_position);
  out.profile_velocity = static_cast<float>(profile_velocity);
  out.profile_acceleration = 0.0f;
  return true;
}

// 모터 id(문자열) 목록을 숫자로. 하나라도 잘못되면 빈 벡터.
inline std::vector<uint8_t> parse_motor_ids(const std::vector<std::string> & motor_ids)
{
  std::vector<uint8_t> ids;
  std::set<uint8_t> seen;
  for (const auto & motor_id : motor_ids) {
    const auto id = parse_motor_id(motor_id);
    if (!id || !seen.insert(*id).second) {
      return {};
    }
    ids.push_back(*id);
  }
  return ids;
}

}  // namespace motor_io
