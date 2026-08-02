#include "fsm.hpp"

#include <algorithm>
#include <cmath>

namespace
{
// motor_status(CurrentMotorStatus)/dynamixel_control(DynamixelControlMsgs)의
// 배열은 이 순서(모터 id)대로 채워짐 -> 인덱스 i가 kMotorIdOrder[i]에 대응.
// main_node.cpp/debug_node.cpp에 있던 것과 동일 - fsm.cpp가 상태별 로직을 전부
// 흡수하면서 이 상수들도 여기로 옮김(main_node.cpp/debug_node.cpp가 각자 안
// 들고 있게 해서 하나로 통일 - 예전엔 두 파일의 kOtherArmMotorIds 값이 서로
// 달랐던 버그가 있었음). tilt(23번)는 이 시스템에서 아예 제어 안 하기로 해서
// 목록에서 뺌(다른 시스템이 구동 - 우리가 값을 보내거나 읽을 이유가 없음).
const std::vector<std::string> kMotorIdOrder = {
  "0", "1", "2", "3", "4", "5", "6",
  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
  "22"};

// COMPLETE_PLACE 홈잉이 0.0으로 명령/정지 확인하는 모터들의 kMotorIdOrder 인덱스.
// kMotorIdOrder[0]="0"(l_shoulder_pitch), [2]="2"(l_shoulder_roll), [4]="4"(l_elbow),
// [19]="22"(torso_yaw) - 다리(10~21)는 포함 안 됨(그 자세 그대로 유지).
constexpr std::array<size_t, 4> kArmMotorIndices = {0, 2, 4, 19};

const std::vector<std::string> kLegMotorIds = {
  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"};

// 반대쪽(오른)팔 모터. SIT 모션 종료 시 그 자세에 그대로 고정함.
const std::vector<std::string> kOtherArmMotorIds = {"1", "3", "5"};

const std::string kTorsoMotorId = "22";
const std::string kLeftShoulderPitchMotorId = "0";
const std::string kLeftShoulderRollMotorId = "2";
const std::string kLeftElbowMotorId = "4";
const std::string kGripMotorId = "6";
}  // namespace

ManipulationFSM::ManipulationFSM(const std::string & urdf_path, const std::string & ee_frame)
: phase_(Phase::PICK),
  pick_state_(PickState::SIT),
  place_state_(PlaceState::VIRTUAL_PLACE),
  robot_model_(urdf_path, ee_frame),
  collision_checker_(robot_model_),
  q_(Eigen::VectorXd::Zero(robot_model_.model().nq)),
  prev_solve_phase_(phase_),
  prev_solve_pick_state_(pick_state_),
  prev_solve_place_state_(place_state_)
{
}

void ManipulationFSM::configure_ik(const IkTuning & tuning)
{
  ik_tuning_ = tuning;
  gripper_position_ = tuning.gripper_open_rad;
}

bool ManipulationFSM::advance()
{
  if (phase_ == Phase::PICK) {
    switch (pick_state_) {
      case PickState::SIT:
        pick_state_ = PickState::PICK_READY;
        return true;
      case PickState::PICK_READY:
        pick_state_ = PickState::PICK;
        return true;
      case PickState::PICK:
        pick_state_ = PickState::COMPLETE_GRIP;
        return true;
      case PickState::COMPLETE_GRIP:
        pick_state_ = PickState::DONE;
        return true;
      case PickState::DONE:
        phase_ = Phase::PLACE;
        place_state_ = PlaceState::VIRTUAL_PLACE;
        return true;
    }
  } else {
    switch (place_state_) {
      case PlaceState::VIRTUAL_PLACE:
        place_state_ = PlaceState::COMPLETE_PLACE;
        return true;
      case PlaceState::COMPLETE_PLACE:
        return false;
    }
  }
  return false;
}

void ManipulationFSM::reset()
{
  phase_ = Phase::PICK;
  pick_state_ = PickState::SIT;
  place_state_ = PlaceState::VIRTUAL_PLACE;
  grip_motor_was_moving_ = false;
  motor_goal_positions_.clear();
}

Phase ManipulationFSM::phase() const
{
  return phase_;
}

PickState ManipulationFSM::pick_state() const
{
  return pick_state_;
}

PlaceState ManipulationFSM::place_state() const
{
  return place_state_;
}

bool ManipulationFSM::is_sequence_complete() const
{
  return phase_ == Phase::PLACE && place_state_ == PlaceState::COMPLETE_PLACE;
}

//해쉬테이블에 모터 하나씩 등록함
void ManipulationFSM::set_motor_position(const std::string & motor_id, double position)
{
  motor_positions_[motor_id] = position;
}
void ManipulationFSM::set_motor_velocity(const std::string & motor_id, double velocity)
{
  motor_velocities_[motor_id] = velocity;
}
double ManipulationFSM::motor_position(const std::string & motor_id) const
{
  auto it = motor_positions_.find(motor_id);
  return it == motor_positions_.end() ? 0.0 : it->second;
}
void ManipulationFSM::set_motor_goal_position(const std::string & motor_id, double goal_position)
{
  motor_goal_positions_[motor_id] = goal_position;
}
double ManipulationFSM::motor_goal_position(const std::string & motor_id) const
{
  auto it = motor_goal_positions_.find(motor_id);
  return it == motor_goal_positions_.end() ? motor_position(motor_id) : it->second;
}





void ManipulationFSM::update_vision(double x, double y, double z)
{
  has_pending_vision_ = true;
  vision_x_ = x;
  vision_y_ = y;
  vision_z_ = z;
}

//모션 앉을 때와 일어날 때 모션 번호를 반환하면 fsm 객체에 저장됨
std::optional<int32_t> ManipulationFSM::entry_motion_number() const
{
  if (phase_ == Phase::PICK && pick_state_ == PickState::SIT) {
    return 73;
  }
  if (phase_ == Phase::PICK && pick_state_ == PickState::COMPLETE_GRIP) {
    return 74;
  }
  return std::nullopt;
}



//fsm 객체에 string 값 저장용
std::string ManipulationFSM::to_string() const
{
  if (phase_ == Phase::PICK) {
    return "PICK." + pick_state_name(pick_state_);
  }
  return "PLACE." + place_state_name(place_state_);
}
std::string ManipulationFSM::pick_state_name(PickState s)
{
  switch (s) {
    case PickState::SIT: return "SIT";
    case PickState::PICK_READY: return "PICK_READY";
    case PickState::PICK: return "PICK";
    case PickState::COMPLETE_GRIP: return "COMPLETE_GRIP";
    case PickState::DONE: return "DONE";
  }
  return "UNKNOWN";
}
std::string ManipulationFSM::place_state_name(PlaceState s)
{
  switch (s) {
    case PlaceState::VIRTUAL_PLACE: return "VIRTUAL_PLACE";
    case PlaceState::COMPLETE_PLACE: return "COMPLETE_PLACE";
  }
  return "UNKNOWN";
}



// action_(재사용되는 멤버)을 갱신하고 참조를 돌려줌 - 상태 필드는 항상 지금 값으로,
// 나머지는 인자로 받은 값 그대로.
FsmAction & ManipulationFSM::makeActionSnapshot(bool publish_motor_command, double profile_velocity,
                                                 std::optional<int32_t> play_motion_number,
                                                 bool deactivate_and_notify)
{
  action_.publish_motor_command = publish_motor_command;
  action_.profile_velocity = profile_velocity;
  action_.play_motion_number = play_motion_number;
  action_.deactivate_and_notify = deactivate_and_notify;
  action_.phase = phase_;
  action_.pick_state = pick_state_;
  action_.place_state = place_state_;
  action_.state_string = to_string();
  return action_;
}

// publish_motor_command가 true면 motor_goal_positions_ 스냅샷을 채워서 action_을 반환.
FsmAction ManipulationFSM::finalize()
{
  if (action_.publish_motor_command) {
    action_.motor_goal_positions = motor_goal_positions_;
  }
  return action_;
}

// 특정 모터가 목표 위치에 도달하고 속도가 충분히 낮은지 판단.
bool ManipulationFSM::isSettled(const std::string & motor_id, double target_rad) const
{
  auto pos_it = motor_positions_.find(motor_id);
  auto vel_it = motor_velocities_.find(motor_id);
  if (pos_it == motor_positions_.end() || vel_it == motor_velocities_.end()) {
    return false;
  }
  return std::fabs(vel_it->second) < ik_tuning_.settle_velocity_threshold &&
         std::fabs(pos_it->second - target_rad) < ik_tuning_.settle_position_tolerance_rad;
}


// 여러 모터가 각자의 목표 위치에 전부 도달했는지 판단(isSettled를 목록에 대해 AND).
// 허리+어깨 회전 settle, COMPLETE_PLACE 홈잉 settle이 전부 이 하나로 통일됨.
bool ManipulationFSM::allSettled(const std::vector<std::pair<std::string, double>> & targets) const
{
  for (const auto & [motor_id, target_rad] : targets) {
    if (!isSettled(motor_id, target_rad)) {
      return false;
    }
  }
  return true;
}

// 그리퍼 역할의 모터가 움직이는 명령어를 주면 움직인다는 것을 이 함수가 판단하고 공을 집었을 때 정지하며 zero position 으로 안 갔는지 확인 zero position으로 가면 공을 
// 못 집은 것으로 판단하고 PICK_READY로 되돌아가게 함. 공을 잡으면 현재 위치를 목표로 고정하고 COMPLETE_GRIP로 전이함
bool ManipulationFSM::checkGripMotorStopped(double motor6_velocity)
{
  if (!(phase_ == Phase::PICK && pick_state_ == PickState::PICK)) {
    grip_motor_was_moving_ = false;
    return false;
  }

  if (std::fabs(motor6_velocity) >= ik_tuning_.grip_moving_velocity_threshold) {
    grip_motor_was_moving_ = true;
    return false;
  } else if (grip_motor_was_moving_ &&
             std::fabs(motor6_velocity) < ik_tuning_.grip_moving_velocity_threshold) {
    double position = motor_position(kGripMotorId);
    if (std::fabs(position) <= ik_tuning_.grip_near_zero_position_threshold) {
      // 목표(0)까지 다 닫힘 -> 아무것도 못 집음 -> 재시도.
      pick_state_ = PickState::PICK_READY;
      return true;
    }
    // 목표에 못 미치고 멈춤 -> 뭔가를 집음 -> 그립 완료로 전이하고 지금 위치에 고정.
    set_motor_goal_position(kGripMotorId, position);
    return advance();
  }
  return false;
}

// --- 4개 공개 진입점의 실제 구현(공개 wrapper는 fsm.hpp에서 finalize()로 감쌈) ---
//모션 번호 반환 받으며 현재 상태와 모션 번호 멤버 업데이트 함 SIT일 때 한 번만 호출되는 함수임
void ManipulationFSM::onActivatedImpl()
{
  const std::optional<int32_t> motion =
    (phase_ == Phase::PICK && pick_state_ == PickState::SIT) ? entry_motion_number() : std::nullopt;
  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/motion, /*deactivate_and_notify=*/false);
}

//모터 상태 업데이트 받을 때마다 호출함.  현재 상태 업데이트 받고 그립 성공시에는 상태만 업데이트 후 모션 호출해서 일어나기, 그리고 손 폈다 접
void ManipulationFSM::onMotorFeedbackImpl() 
{
  // 그립 정지 판정(PICK 상태에서만 유효).
  if (phase_ == Phase::PICK && pick_state_ == PickState::PICK) {
    const double grip_velocity =
      motor_velocities_.count(kGripMotorId) ? motor_velocities_.at(kGripMotorId) : 0.0;
    if (checkGripMotorStopped(grip_velocity)) {
      // PICK_READY로 되돌아간 경우엔 다음 solve_tick()이 state_changed를 스스로
      // 감지해서 홈복귀+재캡처를 시작하므로 여기서 더 할 일 없음.
      const bool completed_grip = pick_state_ == PickState::COMPLETE_GRIP;
      makeActionSnapshot(/*publish_motor_command=*/completed_grip, /*profile_velocity=*/0.0,
                          /*play_motion_number=*/completed_grip ? entry_motion_number() : std::nullopt,
                          /*deactivate_and_notify=*/false);
      return;
    }
  }

  // 그립 위글 시퀀스(COMPLETE_GRIP 모션(74) 재생 종료 후 2틱에 걸쳐 kick->restore). 공을 확실히 집기 위해서 손을 잠깐 벌리고 다시 닫음
  if (pending_grip_wiggle_) {
    const double goal = grip_wiggle_kicked_ ? cached_grip_position_ :
      ik_tuning_.grip_wiggle_kick_position_rad;
    set_motor_goal_position(kGripMotorId, goal);

    if (!grip_wiggle_kicked_) {
      grip_wiggle_kicked_ = true;
      makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                          /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
      return;
    }
    //이제 그리퍼를 다시 오므림
    pending_grip_wiggle_ = false;
    grip_wiggle_kicked_ = false;
    advance();  // COMPLETE_GRIP -> DONE -> (PLACE/VIRTUAL_PLACE로 곧장 넘어감)
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/true);
    return;
  }

  // COMPLETE_PLACE 홈잉은 이제 solveTickImpl()이 담당(command-once + settle-poll을
  // PICK_READY 회전과 동일한 방식으로 통일하기 위함) - 여기선 더 할 일 없음.
  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::onMotionEndImpl(bool ended)
{
  if (!ended) {
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  if (phase_ == Phase::PICK && pick_state_ == PickState::SIT) {
    if (advance()) {  // SIT -> PICK_READY
      for (const auto & id : kLegMotorIds) {
        set_motor_goal_position(id, motor_position(id));
      }
      for (const auto & id : kOtherArmMotorIds) {
        set_motor_goal_position(id, motor_position(id));
      }
      makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                          /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
      return;
    }
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  if (phase_ == Phase::PICK && pick_state_ == PickState::COMPLETE_GRIP) {
    cached_grip_position_ = motor_position(kGripMotorId);
    pending_grip_wiggle_ = true;
    grip_wiggle_kicked_ = false;
  }
  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::solveTickImpl()
{
  const bool state_changed = phase_ != prev_solve_phase_ ||
    pick_state_ != prev_solve_pick_state_ || place_state_ != prev_solve_place_state_;
  if (state_changed) {
    prev_solve_phase_ = phase_;
    prev_solve_pick_state_ = pick_state_;
    prev_solve_place_state_ = place_state_;
  }

  if (phase_ == Phase::PICK) {
    if (pick_state_ == PickState::PICK_READY) {
      stepPickReady(state_changed);
      return;
    }
    if (pick_state_ == PickState::PICK) {
      if (state_changed) {
        damping_ = ik_tuning_.default_damping;
        stuck_streak_ = 0;
        torso_kick_index_ = 0;
        kick_settling_ = false;
        ik_converged_ = false;
        gripper_position_ = ik_tuning_.gripper_open_rad;

        // q_/target_pos_는 유지 - PICK_READY에서 회전시켜둔 자세 + 캡처해둔
        // 목표를 그대로 IK 시작점/목표로 씀(이번 마일스톤의 핵심 요구사항).
      }
      stepReachableIk(/*opening_gripper=*/false);
      return;
    }
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    // SIT / COMPLETE_GRIP / DONE: solve_tick에서 할 일 없음.
    return;
  }

  if (place_state_ == PlaceState::VIRTUAL_PLACE) {
    if (state_changed) {
      damping_ = ik_tuning_.default_damping;
      stuck_streak_ = 0;
      torso_kick_index_ = 0;
      kick_settling_ = false;
      ik_converged_ = false;
      has_captured_target_ = false;
      // gripper_closed_rad(yaml 설정값)이 아니라 실제로 물건을 집은 위치
      // (cached_grip_position_, COMPLETE_GRIP 모션 종료 시점에 캡처해둔 실측값)에서
      // 열기 시작해야 함 - 물건 두께에 따라 실제 그립 위치는 gripper_closed_rad와 다를 수 있음.
      gripper_position_ = cached_grip_position_;
      // q_는 유지(PICK에서 물건을 잡은 자세 그대로 이어감). 회전 준비 단계
      // 없이 비전을 캡처하는 즉시 도달 가능 IK로 들어감("공 놓으면 끝" 요구사항).
    }
    if (!has_captured_target_) {
      if (!has_pending_vision_) {
        makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                            /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
        return;
      }
      target_pos_ = Eigen::Vector3d(vision_x_, vision_y_, vision_z_);
      has_captured_target_ = true;
      // 이 값은 이미 소비했으니, 다음 캡처가 새 update_vision() 없이 이 값을
      // 재사용하지 않도록 리셋함(PICK_READY 캡처와 동일한 이유).
      has_pending_vision_ = false;
    }
    stepReachableIk(/*opening_gripper=*/true);
    return;
  }

  if (place_state_ == PlaceState::COMPLETE_PLACE) {
    stepCompletePlaceHoming(state_changed);
    return;
  }

  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::stepCompletePlaceHoming(bool state_changed)
{
  // motor_goal_positions_는 멤버(계속 유지되는 맵)라서 한 번만 0.0으로 세팅해두면
  // 그 뒤 tick들은 다시 set할 필요 없음 - finalize()가 매번 그 값을 그대로
  // 스냅샷해서 실어줌(매 tick 다시 루프 돌면서 set하던 걸 없앰).
  if (state_changed) {
    for (size_t idx : kArmMotorIndices) {
      if (idx < kMotorIdOrder.size()) {
        set_motor_goal_position(kMotorIdOrder[idx], 0.0);
      }
    }
  }

  std::vector<std::pair<std::string, double>> targets;
  for (size_t idx : kArmMotorIndices) {
    if (idx < kMotorIdOrder.size()) {
      targets.emplace_back(kMotorIdOrder[idx], 0.0);
    }
  }
  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                      /*play_motion_number=*/std::nullopt,
                      /*deactivate_and_notify=*/allSettled(targets));
}

void ManipulationFSM::stepPickReady(bool state_changed)
{
  if (state_changed) {
    q_ = ik_tuning_.home_q_full;
    damping_ = ik_tuning_.default_damping;
    stuck_streak_ = 0;
    torso_kick_index_ = 0;
    kick_settling_ = false;
    has_captured_target_ = false;
    rotation_commanded_ = false;
    
    set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
    set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
    set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
    set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);
    // 이번 tick은 홈 복귀 명령만 - 캡처/회전은 다음 tick부터.
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  if (!has_captured_target_) {
    if (!has_pending_vision_) {  // 아직 비전 자체가 없음 - 대기.
      makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                          /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
      return;
    }
    target_pos_ = Eigen::Vector3d(vision_x_, vision_y_, vision_z_);
    has_captured_target_ = true;
    // 이 값은 이미 소비했으니, 다음 캡처(예: 나중 VIRTUAL_PLACE나 그립 실패 후 재진입)가
    // 새 update_vision() 없이 이 오래된 값을 재사용하지 않도록 반드시 리셋함.
    has_pending_vision_ = false;
  }

  if (!rotation_commanded_) {
    q_[ik::kTorsoJointIndex] = ik_tuning_.torso_target_rad;
    q_[ik::kLeftArmIndices[1]] = ik_tuning_.left_shoulder_roll_target_rad;
    rotation_commanded_ = true;
  }

  set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
  set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);

  const bool settled = allSettled({
      {kTorsoMotorId, ik_tuning_.torso_target_rad},
      {kLeftShoulderRollMotorId, ik_tuning_.left_shoulder_roll_target_rad},
    });
  if (settled) {
    advance();  // PICK_READY -> PICK
  }
  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::stepReachableIk(bool opening_gripper)
{
  if (!ik_converged_) {
    if (kick_settling_) {
      // 킥 직후: 모터가 실제로 킥된 자세(torso+왼팔)에 도달할 때까지 IK 계산을
      // 멈추고 기다림 - 아직 안 도달했는데 그 q_를 시작점으로 계속 IK를 이어가면
      // 실제 로봇 상태와 어긋난 채로 계산하게 됨(PICK_READY 회전 대기와 동일 이유).
      const bool settled = allSettled({
          {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
          {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
          {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
          {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
        });
      if (settled) {
        kick_settling_ = false;
      }
    } else {
      const Eigen::VectorXd secondary_dq = collision_checker_.elbowTorsoAvoidanceDq(
        q_, ik_tuning_.collision_avoidance_threshold_m, ik_tuning_.collision_avoidance_k_pull);
      const bool secondary_active = secondary_dq.squaredNorm() > 0.0;

      ik::IkStepParams params;
      params.tol = ik_tuning_.tol;
      params.alpha = ik_tuning_.alpha;
      params.min_damping = ik_tuning_.min_damping;
      params.max_damping = ik_tuning_.max_damping;
      params.damping_decrease_factor = ik_tuning_.damping_decrease_factor;
      params.damping_increase_factor = ik_tuning_.damping_increase_factor;
      params.joint_weights = ik_tuning_.joint_weights;
      params.joint_weight_scale = ik_tuning_.joint_weight_scale;
      params.secondary_dq = secondary_dq;
      params.secondary_gain = secondary_active ? ik_tuning_.secondary_gain : 0.0;
      params.trust_region_good_ratio = ik_tuning_.trust_region_good_ratio;
      params.trust_region_acceptable_ratio = ik_tuning_.trust_region_acceptable_ratio;

      const ik::CollisionCheckFn collision_fn = [this](const Eigen::VectorXd & q_candidate) {
          return collision_checker_.isColliding(q_candidate);
        };
      const ik::IkStepResult result =
        ik::ikStep(robot_model_, target_pos_, q_, damping_, params, collision_fn);
      q_ = result.q;
      damping_ = result.damping;

      if (result.hard_rejected) {
        ++stuck_streak_;
      } else {
        stuck_streak_ = std::max(0, stuck_streak_ - 1);
      }
      if (stuck_streak_ >= ik_tuning_.stuck_streak_ticks &&
        !ik_tuning_.torso_kick_offsets.empty())
      {
        const double kick =
          ik_tuning_.torso_kick_offsets[torso_kick_index_ % ik_tuning_.torso_kick_offsets.size()];
        ++torso_kick_index_;
        q_ = Eigen::VectorXd::Zero(robot_model_.model().nq);
        q_[ik::kTorsoJointIndex] = kick;
        q_[ik::kLeftArmIndices[1]] = ik_tuning_.left_shoulder_roll_target_rad;
        damping_ = ik_tuning_.default_damping;
        stuck_streak_ = 0;
        kick_settling_ = true;
      }

      if (result.converged) {
        ik_converged_ = true;
      }
    }
  } else if (opening_gripper) {
    gripper_position_ =
      std::min(ik_tuning_.gripper_open_rad, gripper_position_ + ik_tuning_.gripper_step_rad);
    if (gripper_position_ >= ik_tuning_.gripper_open_rad) {
      advance();  // VIRTUAL_PLACE -> COMPLETE_PLACE
    }
  } else {
    gripper_position_ =
      std::max(ik_tuning_.gripper_closed_rad, gripper_position_ - ik_tuning_.gripper_step_rad);
  }

  set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
  set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
  set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
  set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);
  set_motor_goal_position(kGripMotorId, gripper_position_);
  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}
