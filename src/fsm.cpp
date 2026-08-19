#include "fsm.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

namespace
{
const std::vector<std::string> kLegMotorIds = {
  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"};

// 반대쪽(오른)팔 모터. SIT 모션 종료 시 그 자세에 그대로 고정함.
const std::vector<std::string> kOtherArmMotorIds = {"1", "3", "5"};

const std::string kTorsoMotorId = "22";
const std::string kLeftShoulderPitchMotorId = "0";
const std::string kLeftShoulderRollMotorId = "2";
const std::string kLeftElbowMotorId = "4";
const std::string kGripMotorId = "6";

// 비전 쪽이 탐지 실패를 알리는 통짜 센티널 값(x/y/z 셋 다 이 값) -
// tryCapturePendingVision() 참고.
constexpr double kVisionInvalidSentinel = -0.999;
// kVisionInvalidSentinel과의 비교는 exact ==가 아니라 이 오차 안에서 판정함 -
// /master2mani가 geometry_msgs::msg::Point32(필드가 float32)라 -0.999f가 double로
// 승격되면 -0.9990000128746033... 처럼 double 리터럴 -0.999와 비트가 안 맞아서
// exact ==가 항상 false로 새는 버그가 있었음(float32 round-trip 오차는 1e-7 수준 -
// 이 값은 넉넉히 그보다 크고, 실측 좌표가 우연히 -0.999 근처로 올 걱정은 없는 크기).
constexpr double kVisionSentinelEpsilon = 1e-4;
// 공(PICK_READY) 캡처 전용 범위 제한(m) - tryCapturePendingVision() 참고.
constexpr double kVisionMagnitudeBound = 0.33;

// PICK 전용(stepPickIk 참고): EE가 목표 근처까지 왔는데(이 거리 안) 아직
// tol 안으로 수렴은 못 했고, 그런데도 실측 관절 속도(motor_velocity, 시뮬레이션
// dq가 아니라 실제 피드백)가 전부 이 값 미만이면(=사실상 멈춘 상태) joint_weight_scale
// 정규화가 남은 오차를 줄이는 걸 방해한다고 보고 0.0으로 꺼버림.
constexpr double kPickNearTargetDistanceM = 0.02;
constexpr double kPickStalledJointVelocityRadPerSec = 0.1;

// VIRTUAL_PLACE 전용 place_tol_ 초기값(m)과 조정 연속 횟수 기준 - 조정 폭 자체는
// IkTuning::place_tol_increase_step_m/decrease_step_m(main_node.yaml에서 설정) 참고.
constexpr double kPlaceTolInitialM = 0.003;
constexpr int kPlaceTolAdjustStreakTicks = 10;
// place_tol_이 아무리 계속 못 수렴해도 이 이상은 안 풀어줌(관절 한계 등으로 영영
// 도달 불가능한 목표를 만나도 오차 허용치가 끝없이 커지는 걸 막는 안전판).
constexpr double kPlaceTolMaxM = 0.10;
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

Eigen::Vector3d ManipulationFSM::target_pos() const
{
  return target_pos_;
}

std::vector<Eigen::Vector3d> ManipulationFSM::sample_active_trajectory(int num_samples) const
{
  std::vector<Eigen::Vector3d> samples;
  const ik::TrajectoryGenerator * traj = nullptr;
  double duration = 0.0;
  if (phase_ == Phase::PICK && pick_state_ == PickState::PICK) {
    traj = &pick_approach_trajectory_;
    duration = ik_tuning_.pick_approach_duration_sec;
  } else if (phase_ == Phase::PLACE && place_state_ == PlaceState::VIRTUAL_PLACE) {
    traj = &place_trajectory_;
    duration = ik_tuning_.place_traj_duration_sec;
  } else {
    return samples;  // 그 외 상태는 활성 궤적이 없음 - 빈 벡터.
  }
  if (num_samples < 2) {
    return samples;
  }
  samples.reserve(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    const double t = duration * static_cast<double>(i) / static_cast<double>(num_samples - 1);
    samples.push_back(traj->result(t));
  }
  return samples;
}

//해쉬테이블에 모터 하나씩 등록함
void ManipulationFSM::set_motor_position(const std::string & motor_id, double position)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  motor_positions_[motor_id] = position;
}
void ManipulationFSM::set_motor_velocity(const std::string & motor_id, double velocity)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  motor_velocities_[motor_id] = velocity;
}
double ManipulationFSM::motor_position(const std::string & motor_id) const
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = motor_positions_.find(motor_id);
  return it == motor_positions_.end() ? 0.0 : it->second;
}
double ManipulationFSM::motor_velocity(const std::string & motor_id) const
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = motor_velocities_.find(motor_id);
  return it == motor_velocities_.end() ? 0.0 : it->second;
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

void ManipulationFSM::set_tilt_position(double position)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  tilt_position_ = position;
}
void ManipulationFSM::set_tilt_velocity(double velocity)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  tilt_velocity_ = velocity;
}





void ManipulationFSM::update_vision(double x, double y, double z)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  has_pending_vision_ = true;
  vision_x_ = x;
  vision_y_ = y;
  vision_z_ = z;
}

bool ManipulationFSM::tryCapturePendingVision(Eigen::Vector3d * out, bool check_magnitude_bound)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (!has_pending_vision_) {
    return false;
  }
  has_pending_vision_ = false;
  // 비전 쪽이 탐지 실패를 x/y/z 셋 다 0.0인 통짜 센티널로 보내는 경우 - 평균에 안
  // 섞이게 이 샘플은 버림(has_pending_vision_은 이미 위에서 소비했으므로 다음
  // 메시지가 와야 재시도됨, count/sum은 안 건드림).
  const bool is_all_zero = vision_x_ == 0.0 && vision_y_ == 0.0 && vision_z_ == 0.0;
  // -0.999(kVisionInvalidSentinel)는 x/y/z가 항상 셋 다 같이 오는 게 아니라 축별로
  // 따로 실패할 수 있음(예: z만 깊이 계산 실패로 -0.999, x/y는 정상)이라 하나라도
  // 센티널이면 샘플 전체를 버림. 비교는 exact ==가 아니라 kVisionSentinelEpsilon
  // 오차 안에서 판정(위 주석 참고 - float32->double 승격 오차로 exact ==는 항상
  // false로 새서 셋 다 -0.999인 표본조차 안 걸러지던 버그가 실측에서 확인됨).
  const bool has_invalid_sentinel =
    std::fabs(vision_x_ - kVisionInvalidSentinel) < kVisionSentinelEpsilon ||
    std::fabs(vision_y_ - kVisionInvalidSentinel) < kVisionSentinelEpsilon ||
    std::fabs(vision_z_ - kVisionInvalidSentinel) < kVisionSentinelEpsilon;
  // 공(PICK_READY) 캡처 전용: x/y/z 셋 다 절댓값이 kVisionMagnitudeBound(0.33)를
  // 넘는 표본도 버림(골대/VIRTUAL_PLACE 캡처는 check_magnitude_bound=false로
  // 호출해서 이 범위 제한을 안 걺).
  const bool is_all_out_of_bound = check_magnitude_bound &&
    std::fabs(vision_x_) > kVisionMagnitudeBound &&
    std::fabs(vision_y_) > kVisionMagnitudeBound &&
    std::fabs(vision_z_) > kVisionMagnitudeBound;
  if (is_all_zero || has_invalid_sentinel || is_all_out_of_bound) {
    return false;
  }
  const Eigen::Vector3d sample(vision_x_, vision_y_, vision_z_);
  vision_sample_sum_ += sample;
  vision_samples_.push_back(sample);
  ++vision_sample_count_;
  if (vision_sample_count_ < kVisionSampleTarget) {
    // 아직 kVisionSampleTarget개를 다 못 모음 - 호출부는 다음 tick에 새 pending
    // 값이 들어오면 다시 시도함(기존 재시도 패턴 그대로, 여기서 블로킹 안 함).
    return false;
  }
  *out = vision_sample_sum_ / static_cast<double>(kVisionSampleTarget);
  for (int i = 0; i < static_cast<int>(vision_samples_.size()); ++i) {
    std::printf(
      "[vision_sample_debug] sample[%d]=(%.4f, %.4f, %.4f)\n",
      i, vision_samples_[i].x(), vision_samples_[i].y(), vision_samples_[i].z());
  }
  vision_sample_count_ = 0;
  vision_sample_sum_.setZero();
  vision_samples_.clear();
  return true;
}

void ManipulationFSM::resetVisionSampleAccumulator()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  vision_sample_count_ = 0;
  vision_sample_sum_.setZero();
  vision_samples_.clear();
}

void ManipulationFSM::discardPendingVision()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  has_pending_vision_ = false;
}

void ManipulationFSM::notify_motion_end(bool ended)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  pending_motion_end_ = ended;
}

std::optional<bool> ManipulationFSM::take_pending_motion_end()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  const std::optional<bool> result = pending_motion_end_;
  pending_motion_end_.reset();
  return result;
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
  if (phase_ == Phase::PICK && pick_state_ == PickState::DONE) {
    return 75;
  }
  if (phase_ == Phase::PLACE && place_state_ == PlaceState::COMPLETE_PLACE) {
    return 76;
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
  if (play_motion_number.has_value()) {
    motion_in_flight_ = true;
  }
  action_.publish_motor_command = publish_motor_command;
  action_.profile_velocity = profile_velocity;
  action_.play_motion_number = play_motion_number;
  action_.deactivate_and_notify = deactivate_and_notify;
  action_.reset_pan_tilt_level = false;  // onMotionEndImpl의 DONE 분기에서만 별도로 true로 덮어씀
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

// 특정 모터가 목표 위치에 도달하고 속도가 충분히 낮은지 판단. motor_positions_/
// motor_velocities_는 motor_status 콜백 스레드가 쓰므로 data_mutex_로 짧게 보호.
bool ManipulationFSM::isSettled(const std::string & motor_id, double target_rad,
                                 std::optional<double> velocity_threshold_override,
                                 std::optional<double> position_tolerance_override) const
{
  // 모션 재생 중엔 외부 모션 재생기가 모터를 직접 구동해서 우리가 세팅해둔
  // target_rad가 더 이상 의미가 없음 - 무조건 미도달로 취급.
  if (motion_in_flight_) {
    return false;
  }

  double position = 0.0;
  double velocity = 0.0;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    auto pos_it = motor_positions_.find(motor_id);
    auto vel_it = motor_velocities_.find(motor_id);
    if (pos_it == motor_positions_.end() || vel_it == motor_velocities_.end()) {
      return false;
    }
    position = pos_it->second;
    velocity = vel_it->second;
  }
  const double velocity_threshold =
    velocity_threshold_override.value_or(ik_tuning_.settle_velocity_threshold);
  const double position_tolerance_rad =
    position_tolerance_override.value_or(ik_tuning_.settle_position_tolerance_rad);
  return std::fabs(velocity) < velocity_threshold &&
         std::fabs(position - target_rad) < position_tolerance_rad;
}


// 여러 모터가 각자의 목표 위치에 전부 도달했는지 판단(isSettled를 목록에 대해 AND).
// 허리+어깨 회전 settle, COMPLETE_PLACE 홈잉 settle이 전부 이 하나로 통일됨.
bool ManipulationFSM::allSettled(const std::vector<std::pair<std::string, double>> & targets,
                                  std::optional<double> velocity_threshold_override,
                                  std::optional<double> position_tolerance_override) const
{
  for (const auto & [motor_id, target_rad] : targets) {
    if (!isSettled(motor_id, target_rad, velocity_threshold_override,
        position_tolerance_override))
    {
      return false;
    }
  }
  return true;
}

// isSettled()와 동일한 속도/위치 임계값 판정을 tilt 전용으로 - 목표는 레벨(0.0) 고정.
bool ManipulationFSM::isTiltSettled() const
{
  double position = 0.0;
  double velocity = 0.0;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    position = tilt_position_;
    velocity = tilt_velocity_;
  }
  return std::fabs(velocity) < ik_tuning_.settle_velocity_threshold &&
         std::fabs(position) < ik_tuning_.settle_position_tolerance_rad;
}

// 그리퍼 역할의 모터가 움직이는 명령어를 주면 움직인다는 것을 이 함수가 판단하고 공을 집었을 때 정지하며 완전히 닫힘(gripper_closed_rad, 아무것도 못 집었을 때
// 도달하는 자리 - 0rad이 아님, config 참고)까지 갔는지 확인. 그 자리까지 갔으면 공을
// 못 집은 것으로 판단하고 PICK_READY로 되돌아가게 함. 공을 잡으면 현재 위치를 목표로 고정하고 COMPLETE_GRIP로 전이함
bool ManipulationFSM::checkGripMotorStopped(double motor6_velocity)
{
  // ik_converged_까지 확인해야 함 - PICK_READY 동안 그리퍼가 closed_rad에 그대로
  // 있다가(stepPickReady는 그리퍼를 안 건드림) PICK 재진입 시 gripper_position_이
  // open으로 리셋되면서 그리퍼가 다시 열리는 움직임 자체가 "그립 모터가 움직였다가
  // 멈춤"으로 잡혀서, IK가 수렴하기도 전에(아직 실제로 닫으라고 명령도 안 냈는데)
  // "멈춘 위치가 closed_rad 근처가 아니다" -> "뭔가 집었다"로 오판하는 버그가 있었음.
  // IK가 수렴해서 실제로 닫으라는 명령이 나간 뒤부터만 이 판정을 시작함.
  if (!(phase_ == Phase::PICK && pick_state_ == PickState::PICK && ik_converged_)) {
    grip_motor_was_moving_ = false;
    grip_stopped_streak_ = 0;
    return false;
  }

  // grip_motor_was_moving_: 그립 모터가 최소 한 번은 실제로 빠르게 움직인 적이
  // 있어야만(닫힘 명령을 받고 움직이기 시작한 뒤여야만) "멈췄다" 판정을 시작함 -
  // 안 그러면 닫힘 명령 전 가만히 서있는 상태(velocity=0)를 곧바로 "멈춤"으로 오판함.
  // grip_stopped_streak_: velocity가 threshold 밑으로 연속 몇 tick째인지 셈 - 1틱만
  // 보고 확정하면 정지 마찰/토크 리플 같은 순간적인 속도 저하를 "완전히 멈췄다"로
  // 오판해서 훨씬 덜 오므린 위치에서 그립 완료로 확정돼버리는 문제가 있었음.
  const bool is_moving_fast =
    std::fabs(motor6_velocity) >= ik_tuning_.grip_moving_velocity_threshold;
  grip_motor_was_moving_ = grip_motor_was_moving_ || is_moving_fast;
  grip_stopped_streak_ = is_moving_fast ? 0 : grip_stopped_streak_ + 1;

  if (grip_motor_was_moving_ && grip_stopped_streak_ > 0) {
    const double position = motor_position(kGripMotorId);
    std::printf(
      "[grip_stop_debug] streak=%d/%d position=%.4f closed_rad=%.4f diff=%.4f\n",
      grip_stopped_streak_, ik_tuning_.grip_stopped_streak_ticks, position,
      ik_tuning_.gripper_closed_rad, position - ik_tuning_.gripper_closed_rad);
  }

  const bool confirmed_stopped =
    grip_motor_was_moving_ && grip_stopped_streak_ >= ik_tuning_.grip_stopped_streak_ticks;
  if (!confirmed_stopped) {
    return false;
  }

  const double position = motor_position(kGripMotorId);
  if (std::fabs(position - ik_tuning_.gripper_closed_rad) <=
      ik_tuning_.grip_near_zero_position_threshold) {
    // 완전히 닫힘(gripper_closed_rad)까지 다 감 -> 아무것도 못 집음 -> 재시도.
    pick_state_ = PickState::PICK_READY;
    grip_motor_was_moving_ = false;
    grip_stopped_streak_ = 0;
    return true;
  }

  // 목표에 못 미치고 멈춤 -> 뭔가를 집음 -> 그립 완료로 전이. goal은 그 순간
  // 실측 position이 아니라 gripper_closed_rad(yaml 설정값)로 고정 - 물체를 계속
  // 완전히 닫히는 방향으로 밀어붙여야(토크로 유지) 안정적으로 잡고 있음.
  set_motor_goal_position(kGripMotorId, ik_tuning_.gripper_closed_rad);
  grip_motor_was_moving_ = false;
  grip_stopped_streak_ = 0;
  return advance();
}

// --- 4개 공개 진입점의 실제 구현(공개 wrapper는 fsm.hpp에서 finalize()로 감쌈) ---
//모션 번호 반환 받으며 현재 상태와 모션 번호 멤버 업데이트 함 SIT일 때 한 번만 호출되는 함수임
void ManipulationFSM::onActivatedImpl()
{
  // SIT(픽 시퀀스 첫 활성화)뿐 아니라 DONE(픽 끝나고 place 하려고 재활성화된 경우)도
  // 진입 모션이 있음 - 그 외 상태에서 재활성화되는 경우는 없음(picking 도중엔
  // deactivate_and_notify가 안 나가므로).
  const bool plays_entry_motion =
    phase_ == Phase::PICK && (pick_state_ == PickState::SIT || pick_state_ == PickState::DONE);
  const std::optional<int32_t> motion = plays_entry_motion ? entry_motion_number() : std::nullopt;
  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/motion, /*deactivate_and_notify=*/false);
}

//  현재 상태 업데이트 받고 그립 성공시에는 상태만 업데이트 후 모션 호출해서 일어나기, 그리고 손 폈다 접
void ManipulationFSM::onMotorFeedbackImpl() 
{
  // 그립 정지 판정(PICK 상태에서만 유효).
  if (phase_ == Phase::PICK && pick_state_ == PickState::PICK) {
    const double grip_velocity = motor_velocity(kGripMotorId);
    if (checkGripMotorStopped(grip_velocity)) {
      // PICK_READY로 되돌아간 경우, COMPLETE_GRIP으로 전이한 경우 둘 다 다음
      // solve_tick()이 state_changed를 스스로 감지해서 처리함(PICK_READY는
      // 홈복귀+재캡처, COMPLETE_GRIP은 왼팔을 홈으로 복귀시키고 settle되면
      // stepCompleteGripHoming()이 모션(74)을 재생함) - 여기서 더 할 일 없음.
      makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                          /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
      return;
    }
  }

  // 그립 위글 시퀀스(COMPLETE_GRIP 모션(74) 재생 종료 후 kick->restore). 공을
  // 확실히 집기 위해서 손을 잠깐 벌렸다가 다시 닫음 - 각 단계는 고정 틱 수가
  // 아니라 isSettled()로 실제 도달을 확인한 뒤에만 다음 단계로 넘어감(다른
  // step*Homing들과 동일한 command-once + settle-poll 패턴).
  if (pending_grip_wiggle_) {
    if (!grip_wiggle_kicked_) {
      set_motor_goal_position(kGripMotorId, ik_tuning_.grip_wiggle_kick_position_rad);
      // 그립 위글(그리퍼만 살짝 열었다 닫기) 동안 왼팔이 흔들리지 않게, 이 순간의
      // 실제 위치를 그대로 goal로 고정해둠(다리/반대팔을 고정하던 것과 동일 패턴).
      set_motor_goal_position(kTorsoMotorId, motor_position(kTorsoMotorId));
      set_motor_goal_position(kLeftShoulderPitchMotorId, motor_position(kLeftShoulderPitchMotorId));
      set_motor_goal_position(kLeftShoulderRollMotorId, motor_position(kLeftShoulderRollMotorId));
      set_motor_goal_position(kLeftElbowMotorId, motor_position(kLeftElbowMotorId));
      grip_wiggle_kicked_ = true;
    } else if (!grip_wiggle_restore_commanded_) {
      if (isSettled(kGripMotorId, ik_tuning_.grip_wiggle_kick_position_rad)) {
        // 위글 복원 목표는 cached_grip_position_(그 순간 실측값)이 아니라 그냥
        // gripper_closed_rad(yaml 설정값)로 고정 - 위글은 "실제로 다시 오므려
        // 잡았는지"만 확인하면 되므로 매번 같은 기준값을 씀.
        set_motor_goal_position(kGripMotorId, ik_tuning_.gripper_closed_rad);
        grip_wiggle_restore_commanded_ = true;
        grip_wiggle_restore_wait_ticks_ = 0;
      }
    } else {
      ++grip_wiggle_restore_wait_ticks_;
      // 그리퍼가 물체를 물어서 gripper_closed_rad까지 실측이 못 좁혀지면 isSettled가
      // 영영 안 될 수 있어서, grip_wiggle_restore_timeout_ticks 틱이 지나면 settle
      // 여부와 무관하게 강제로 넘어감(무한 대기 방지).
      if (isSettled(kGripMotorId, ik_tuning_.gripper_closed_rad) ||
        grip_wiggle_restore_wait_ticks_ > ik_tuning_.grip_wiggle_restore_timeout_ticks)
      {
        //이제 그리퍼가 실제로 다시 오므려진 걸 확인함(또는 타임아웃으로 강제 진행)
        pending_grip_wiggle_ = false;
        grip_wiggle_kicked_ = false;
        grip_wiggle_restore_commanded_ = false;
        advance();  // COMPLETE_GRIP -> DONE
        // 곧바로 deactivate하지 않음 - solveTickImpl()의 stepDoneHoming()이 왼팔을
        // 한 번 더 home_q로 복귀시키고 settle까지 확인한 뒤에 deactivate함.
      }
    }
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  // COMPLETE_PLACE 홈잉은 이제 solveTickImpl()이 담당(command-once + settle-poll을
  // PICK_READY 회전과 동일한 방식으로 통일하기 위함) - 여기선 더 할 일 없음.
  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::onMotionEndImpl(bool ended)
{
  motion_in_flight_ = false;

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
      gripper_position_ = ik_tuning_.gripper_open_rad;
      set_motor_goal_position(kGripMotorId, gripper_position_);
      makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                          /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
      return;
    }
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  if (phase_ == Phase::PICK && pick_state_ == PickState::DONE) {
    // 모션 75(DONE 재활성화 진입 모션)가 끝나는 순간 - advance()가 상태를 바꾸기
    // 전에 여기서 판정. 카메라를 레벨(tilt=0)로 되돌리라고 main_node.cpp에 알림.
    if (advance()) {  // DONE -> PLACE.VIRTUAL_PLACE
      for (const auto & id : kLegMotorIds) {
        set_motor_goal_position(id, motor_position(id));
      }
      for (const auto & id : kOtherArmMotorIds) {
        set_motor_goal_position(id, motor_position(id));
      }
      // 다리/반대팔뿐 아니라 왼팔(torso+3관절)/그립까지 전부 이 순간 실측 위치로
      // 고정 - 이 tick의 스냅샷이 motor_goal_positions_에 값이 없던(갱신 안 된)
      // 모터를 이전 상태의 stale한 목표로 내보내는 걸 막음.
      set_motor_goal_position(kTorsoMotorId, motor_position(kTorsoMotorId));
      set_motor_goal_position(kLeftShoulderPitchMotorId, motor_position(kLeftShoulderPitchMotorId));
      set_motor_goal_position(kLeftShoulderRollMotorId, motor_position(kLeftShoulderRollMotorId));
      set_motor_goal_position(kLeftElbowMotorId, motor_position(kLeftElbowMotorId));
      set_motor_goal_position(kGripMotorId, motor_position(kGripMotorId));
      makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                          /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
      action_.reset_pan_tilt_level = true;
      return;
    }
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    action_.reset_pan_tilt_level = true;
    return;
  }

  if (phase_ == Phase::PICK && pick_state_ == PickState::COMPLETE_GRIP) {
    cached_grip_position_ = motor_position(kGripMotorId);
    pending_grip_wiggle_ = true;
    grip_wiggle_kicked_ = false;
    grip_wiggle_restore_commanded_ = false;
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

  if (phase_ == Phase::PLACE && place_state_ == PlaceState::COMPLETE_PLACE) {
    // stepCompletePlaceHoming이 역재생 IK 수렴 + settle 확인 후 재생한 모션(76)이
    // 끝난 시점 - 시퀀스 전체의 마지막이라 여기서 바로 deactivate. 다른 모션 종료
    // 분기(SIT/DONE/COMPLETE_GRIP)와 동일하게 다리/반대팔은 그 시점 실측 위치로
    // 고정해서 마지막 명령으로 함께 보냄.
    for (const auto & id : kLegMotorIds) {
      set_motor_goal_position(id, motor_position(id));
    }
    for (const auto & id : kOtherArmMotorIds) {
      set_motor_goal_position(id, motor_position(id));
    }
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/true);
    return;
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
        kick_settling_ = false;
        ik_converged_ = false;
        gripper_position_ = ik_tuning_.gripper_open_rad;
        active_joint_weights_ = ik_tuning_.joint_weights;
        active_joint_weight_scale_ = ik_tuning_.joint_weight_scale;

        // q_/target_pos_는 유지 - PICK_READY에서 회전시켜둔 자세 + 캡처해둔
        // 목표를 그대로 IK 시작점/목표로 씀(이번 마일스톤의 핵심 요구사항).
      }
      // 실험적으로 pick_approach_trajectory_ 추적을 꺼봄(그립이 궤적 추적 지연 때문에
      // 엉뚱한 지점에서 일어나는지 확인 중 - 대화 참고) - target_pos_는 state_changed에서
      // 이미 캡처된 최종 목표값 그대로 유지되므로, 궤적 보간 없이 처음부터 그 점을
      // 바로 목표로 씀. 원복하려면 아래 주석을 풀고 이 블록의 target_pos_ 직행 두 줄을
      // 지우면 됨.
      // const bool approach_in_progress =
      //   pick_approach_traj_time_ < ik_tuning_.pick_approach_duration_sec;
      // target_pos_ = pick_approach_trajectory_.result(pick_approach_traj_time_);
      // pick_approach_traj_time_ += ik_tuning_.place_traj_dt_sec;
      // if (approach_in_progress) {
      //   ik_converged_ = false;
      // }
      stepPickIk();
      return;
    }
    if (pick_state_ == PickState::COMPLETE_GRIP) {
      stepCompleteGripHoming(state_changed);
      return;
    }
    if (pick_state_ == PickState::DONE) {
      stepDoneHoming(state_changed);
      return;
    }
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    // SIT: solve_tick에서 할 일 없음.
    return;
  }

  if (place_state_ == PlaceState::VIRTUAL_PLACE) {
    if (state_changed) {
      damping_ = ik_tuning_.default_damping;
      stuck_streak_ = 0;
      // place_tol_ 참고 - 매번 새 VIRTUAL_PLACE 시도는 항상 kPlaceTolInitialM부터 다시 시작.
      place_tol_ = kPlaceTolInitialM;
      place_tol_success_streak_ = 0;
      place_tol_fail_streak_ = 0;
      place_release_shoulder_pitch_commanded_ = false;
      place_release_elbow_commanded_ = false;
      place_release_error_confirmed_ok_ = false;
      kick_settling_ = false;
      ik_converged_ = false;
      has_captured_target_ = false;
      resetVisionSampleAccumulator();
      tilt_settle_vision_discarded_ = false;
      // stepPickReady처럼 여기서 바로 discardPendingVision()을 부르진 않음 - 아직
      // tilt가 settle되기 전이라 discard해봤자 그 사이 계속 들어오는 값으로 다시
      // 채워짐. 대신 isTiltSettled()가 처음 true가 되는 tick에 한 번만 버림(아래).
      active_joint_weights_ = ik_tuning_.place_joint_weights;
      active_joint_weight_scale_ = ik_tuning_.joint_weight_scale;
      place_trajectory_.clear();
      place_traj_time_ = 0.0;
      // gripper_closed_rad(yaml 설정값)이 아니라 실제로 물건을 집은 위치
      // (cached_grip_position_, COMPLETE_GRIP 모션 종료 시점에 캡처해둔 실측값)에서
      // 열기 시작해야 함 - 물건 두께에 따라 실제 그립 위치는 gripper_closed_rad와 다를 수 있음.
      gripper_position_ = cached_grip_position_;
      // q_를 PICK 마지막 IK 결과가 아니라 이 순간 실측 모터 위치로 다시 채움 - PICK
      // 종료 후 모션(75) 재생 등을 거치며 실제 자세가 소프트웨어 q_보다 앞서
      // 나갔을 수 있어서(대화 참고), 그 어긋남이 place_trajectory_ 시작점(FK 기준
      // start_pos)에 그대로 반영되지 않게 함. 회전 준비 단계 없이 비전을 캡처하는
      // 즉시 도달 가능 IK로 들어감("공 놓으면 끝" 요구사항).
      q_[ik::kTorsoJointIndex] = motor_position(kTorsoMotorId);
      q_[ik::kLeftArmIndices[0]] = motor_position(kLeftShoulderPitchMotorId);
      q_[ik::kLeftArmIndices[1]] = motor_position(kLeftShoulderRollMotorId);
      q_[ik::kLeftArmIndices[2]] = motor_position(kLeftElbowMotorId);
    }
    if (!has_captured_target_) {
      // 모션 75 종료 시 카메라를 레벨로 되돌리라고 명령했지만(onMotionEndImpl의
      // reset_pan_tilt_level -> main_node.cpp가 /master/pan_tilt에 place용 tilt
      // 모드=5(0°)를 publish), 실제로 그 각도까지 돌아오는 데는 시간이 걸림 - 아직 안
      // 돌아온 채로 찍은 좌표는 못 믿으므로 settle될 때까지 캡처 시도 자체를 미룸.
      if (!isTiltSettled()) {
        makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                            /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
        return;
      }
      std::printf("[place_tilt_debug] tilt settled\n");
      // 골대(place 목표) 캡처 - 공(PICK_READY)과 달리 magnitude bound는 안 걺.
      if (!tryCapturePendingVision(&target_pos_, /*check_magnitude_bound=*/false)) {
        makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                            /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
        return;
      }
      // 비전 y의 실측 오프셋을 캡처 시점에 바로 보정(stepPickReady와 동일).
      //target_pos_.y() -= ik_tuning_.vision_y_offset;
      has_captured_target_ = true;

      // 시작점(현재 EE, q_로 FK) -> 정점(장애물 회피, 고정 x/y 튜닝값) -> 목표(비전
      // 캡처값) 9점 등록. 각 점의 속도는 더 이상 여기서 손으로 안 찍고
      // TrajectoryGenerator/Trajectory1D가 IPTP식으로 자동 추정함(estimateVelocities()
      // 참고) - 이전에 z만 시작/끝 vel을 일치시켜 등속 직선으로 강제하던 트릭은 그
      // 대가로 포기함(양 끝점 속도가 이제 항상 0으로 고정되므로). z 위치값 자체는
      // 이전과 동일하게 등속 직선 근사(vz)로 계산.
      const Eigen::Vector3d start_pos = ik::eePosition(robot_model_, q_);
      const double duration = ik_tuning_.place_traj_duration_sec;

      const double vz = (target_pos_.z() - start_pos.z()) / duration;

      place_trajectory_.put_point(0.0, start_pos.x(), start_pos.y(), start_pos.z());
      place_trajectory_.put_point(
        duration * 0.1, target_pos_.x()*0.1, target_pos_.y() + ik_tuning_.place_apex_y,
        start_pos.z() + vz * duration * 0.1);
      place_trajectory_.put_point(
        duration * 0.2, target_pos_.x()*0.5, target_pos_.y() + ik_tuning_.place_apex_y,
        start_pos.z() + vz * duration * 0.2);
      place_trajectory_.put_point(
        duration * 0.3, target_pos_.x()*0.8, target_pos_.y() + ik_tuning_.place_apex_y,
        start_pos.z() + vz * duration * 0.3);
      place_trajectory_.put_point(
        duration * 0.5, target_pos_.x()*0.99-0.05, target_pos_.y() + ik_tuning_.place_apex_y,
        start_pos.z() + vz * duration * 0.8);
      place_trajectory_.put_point(
        duration * 0.7, target_pos_.x()-0.08, target_pos_.y() + ik_tuning_.place_apex_y,
        start_pos.z() + vz * duration * 1.0);
      place_trajectory_.put_point(
        duration * 0.8, target_pos_.x()-0.08, target_pos_.y() + ik_tuning_.place_apex_y,
        start_pos.z() + vz * duration +  0.08);
      place_trajectory_.put_point(
        duration * 0.9, target_pos_.x()-0.08, target_pos_.y() + ik_tuning_.place_apex_y-0.05,
        start_pos.z() + vz * duration + 0.08);
      place_trajectory_.put_point(
        duration, target_pos_.x()-0.08, target_pos_.y()+0.05,
        start_pos.z() + vz * duration + 0.08);
    }
    // 궤적이 아직 끝나지 않았으면(목표 지점 도달 전) target_pos_가 매 tick 계속
    // 움직이므로, 이전 tick에 ik_converged_가 true가 됐어도 다시 풀게 리셋함 -
    // 안 그러면 궤적 시작점(=캡처 직전의 현재 EE 위치라 오차가 거의 0)에서 바로
    // "도달함"으로 굳어버려서, 그 뒤로 목표가 정점/최종 목표로 계속 움직여도
    // stepPlaceIk의 if(!ik_converged_) 가드에 막혀 다시는 IK를 안 풀게 됨
    // (그리퍼가 시작 위치에서 바로 열려버리는 버그).

    // place_traj_time_을 흘려보내기(=다음 t로 목표를 옮기기) 전에, 방금 IK가
    // 계산해둔 q_(직전 tick의 결과)에 관절들이 실제로 도달했는지 먼저 확인함 -
    // 안 그러면 실제 로봇이 못 따라오는 동안에도 place_traj_time_/target_pos_는
    // 계속 앞으로 흘러가버려서, 소프트웨어 목표가 실제 팔보다 앞서 나가며 궤적
    // 중간 지점들을 스치듯 지나쳐버림(그리퍼가 팔 도달 전에 열리던 문제와 같은
    // 원인 - kick_settling_과 동일 패턴).
    // if (!allSettled({
    //     {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
    //     {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
    //     {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
    //     {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
    //   }))
    // {
    //   makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
    //                       /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    //   return;
    // }

    const bool trajectory_in_progress = place_traj_time_ < ik_tuning_.place_traj_duration_sec;
    target_pos_ = place_trajectory_.result(place_traj_time_);
    place_traj_time_ += ik_tuning_.place_traj_dt_sec;
    if (trajectory_in_progress) {
      ik_converged_ = false;
    }
    stepPlaceIk(/*opening_gripper=*/true);
    return;
  }

  if (place_state_ == PlaceState::COMPLETE_PLACE) {
    stepCompletePlaceHoming(state_changed);
    return;
  }

  makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

// 그립 성공(COMPLETE_GRIP 진입) 직후: 모션을 바로 재생하지 않고, 먼저 왼팔을
// (집은 채로 안전하게 들고 있을) 별도 자세로 복귀시켜 전부 settle된 뒤에야
// 모션(74)을 재생함(집은 자세 그대로 모션을 태우면 팔이 이상한 궤적을 그릴 수 있어서).
void ManipulationFSM::stepCompleteGripHoming(bool state_changed)
{
  if (state_changed) {
    // 고정 프리셋 관절각으로 바로 스냅하지 않고, 방금 집은 자세(state_changed
    // 이전 q_ - PICK에서 IK로 수렴해둔 그 값)의 EE 위치를 FK로 구한 뒤 z만
    // complete_grip_lift_z_m(기본 10cm)만큼 들어올린 지점을 새 IK 목표로 잡음 -
    // 공을 안전하게 들어올린 자세로 자연스럽게 옮겨감.
    const Eigen::Vector3d grip_ee_pos = ik::eePosition(robot_model_, q_);
    target_pos_ = grip_ee_pos;
    target_pos_.z() += ik_tuning_.complete_grip_lift_z_m;
    damping_ = ik_tuning_.default_damping;
    stuck_streak_ = 0;
    kick_settling_ = false;
    ik_converged_ = false;
    active_joint_weights_ = ik_tuning_.joint_weights;
    // 여기선 joint_weight_scale 정규화를 끔(0.0) - PICK/PLACE처럼 torso 사용을
    // 아껴야 할 이유가 없고(그냥 x로 10cm 당기는 단순 목표), 정규화가 수렴을
    // 방해하지 않게 함.
    active_joint_weight_scale_ = 0.0;
  }

  // 모션(74)이 재생 중이면 SIT/DONE 진입 모션과 동일하게 완전히 조용해짐
  // (publish_motor_command=false) - 모션 재생 중엔 외부 모션 재생기가 이 모터들을
  // 직접 구동하므로 우리가 계속 q_ 유지 명령을 보내면 충돌할 수 있고, isSettled()가
  // motion_in_flight_일 때 무조건 false를 주므로 allSettled도 이 시점엔 항상 false.
  if (motion_in_flight_) {
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  // 당김 목표(target_pos_)로 아직 수렴 안 했으면 stepPickIk가 매 tick 계속
  // damped-IK를 풀고(q_ + set_motor_goal_position까지 안에서 다 처리), 수렴한
  // 뒤에야 아래 settle 확인 + 모션(74) 재생 단계로 넘어감. 그리퍼는 이미
  // PICK에서 닫아뒀고 여기서 안 건드림(stepPickIk는 항상 닫힌 채로 유지).
  if (!ik_converged_) {
    stepPickIk();
    return;
  }

  // 실측으로 이 대기가 15~20초까지 걸리는 걸 확인해서(대화 참고), 다른 settle
  // 판정(허리/어깨 회전 대기 등)엔 영향 없이 이 구간만 더 느슨한 임계값
  // (ik_tuning_.complete_grip_settle_*)을 override로 넘김.
  const bool settled = allSettled({
      {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
      {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
      {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
      {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
    }, ik_tuning_.complete_grip_settle_velocity_threshold,
    ik_tuning_.complete_grip_settle_position_tolerance_rad);

  std::optional<int32_t> motion;
  if (settled) {
    motion = entry_motion_number();
  }

  makeActionSnapshot(/*publish_motor_command=*/!settled, /*profile_velocity=*/settled ? 0.0 : 1.0,
                      /*play_motion_number=*/motion, /*deactivate_and_notify=*/false);
}

// 그립 위글까지 끝나서 DONE에 진입한 직후: 곧바로 deactivate하지 않고, 왼팔을
// 0.0(홈)으로 복귀시켜 전부 settle될 때까지 기다린 뒤에야 deactivate함
// (stepCompletePlaceHoming()과 동일한 command-once + settle-poll 패턴).
void ManipulationFSM::stepDoneHoming(bool state_changed)
{
  if (state_changed) {
    q_ = Eigen::VectorXd::Zero(robot_model_.model().nq);
    set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
    set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
    set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
    set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);
  }

  // DONE 재활성화 시(place 단계 시작) onActivatedImpl()이 모션(75)을 재생하는데,
  // 그동안에도 이 함수는 매 틱 계속 불림 - 모션 재생 중엔 stepCompleteGripHoming과
  // 동일한 이유로 완전히 조용해져야 함(publish_motor_command=false).
  if (motion_in_flight_) {
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  const bool settled = allSettled({
      {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
      {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
      {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
      {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
    });

  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/0.0,
                      /*play_motion_number=*/std::nullopt,
                      /*deactivate_and_notify=*/settled);
}

void ManipulationFSM::stepCompletePlaceHoming(bool state_changed)
{
  if (state_changed) {
    // [실험] place_trajectory_를 IK로 되짚어가는 대신, q_를 ready pose(ik_tuning_.home_q_full,
    // yaml의 pick_ready_home_q)로 직접 스냅하고 실측 settle만 확인함(대화 참고) -
    // 예전엔 그 뒤에 완전 홈(0)으로 한 번 더 스냅하고 다시 settle을 기다리는 2단계가
    // 있었는데, ready pose 도달 확인만으로 충분하다고 판단해서 그 단계는 제거함(대화
    // 참고 - 아래 본문). 기존 IK 역재생 초기화/루프는 그대로 두되 전부 주석 처리해서
    // 언제든 되돌릴 수 있게 함.
    q_[ik::kTorsoJointIndex] = ik_tuning_.home_q_full[ik::kTorsoJointIndex];
    q_[ik::kLeftArmIndices[0]] = ik_tuning_.home_q_full[ik::kLeftArmIndices[0]];
    // left_shoulder_roll은 home_q_full 값(0.3) 대신 1.0으로 고정(대화 참고).
    q_[ik::kLeftArmIndices[1]] = 1.0;
    q_[ik::kLeftArmIndices[2]] = ik_tuning_.home_q_full[ik::kLeftArmIndices[2]];
    set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
    set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
    set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
    set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);

    // 기존 IK 역재생용 초기화(주석 처리 - 대화 참고):
    // // VIRTUAL_PLACE는 그리퍼가 열릴 때까지 기다리는 동안에도 place_traj_time_을
    // // duration 너머로 계속 흘려보내므로(그 tick들도 매번 += dt_sec), 역재생
    // // 시작점을 duration으로 다시 고정하고 IK를 강제로 다시 풀게 함 - place_trajectory_
    // // 자체는 아직 clear() 안 됐으므로(다음 VIRTUAL_PLACE 진입 때만 지워짐) 그대로 재사용.
    // place_traj_time_ = ik_tuning_.place_traj_duration_sec;
    // ik_converged_ = false;
    // damping_ = ik_tuning_.default_damping;
    // stuck_streak_ = 0;
    // place_tol_ = kPlaceTolInitialM;
    // place_tol_success_streak_ = 0;
    // place_tol_fail_streak_ = 0;
    // kick_settling_ = false;
    // active_joint_weights_ = ik_tuning_.complete_place_joint_weights;
    // active_joint_weight_scale_ = ik_tuning_.joint_weight_scale;
  }

  // 모션(76)이 재생 중이면 SIT/DONE/COMPLETE_GRIP 진입 모션과 동일하게 완전히
  // 조용해짐(stepCompleteGripHoming과 동일 이유 - 외부 모션 재생기가 다리/반대팔을
  // 포함한 전신을 직접 구동하는 동안 우리가 q_ 유지 명령을 계속 보내면 충돌함).
  if (motion_in_flight_) {
    makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  // 기존 IK 역재생 루프(주석 처리 - 대화 참고):
  // if (place_traj_time_ > 0.0 || !ik_converged_) {
  //   if (!allSettled({
  //       {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
  //       {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
  //       {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
  //       {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
  //     }))
  //   {
  //     makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
  //                         /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
  //     return;
  //   }
  //   const bool reverse_in_progress = place_traj_time_ > 0.0;
  //   target_pos_ = place_trajectory_.result(place_traj_time_);
  //   place_traj_time_ = std::max(0.0, place_traj_time_ - ik_tuning_.place_traj_dt_sec);
  //   if (reverse_in_progress) {
  //     ik_converged_ = false;
  //   }
  //   stepPlaceIk(/*opening_gripper=*/false);
  //   return;
  // }

  // ready pose(위 state_changed에서 스냅한 q_)에 실측이 도달했는지 확인한 뒤에야
  // (stepCompleteGripHoming과 동일한 command-once + settle-poll 패턴) 모션(76)을
  // 재생해서 전신(다리+반대팔 포함)을 최종 자세로 되돌림 - 모션이 끝나면
  // onMotionEndImpl이 deactivate함. 완전 홈(0)으로 다시 스냅하고 별도로 settle을
  // 기다리던 단계는 필요 없어져서 제거함(대화 참고).
  const bool settled = allSettled({
      {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
      {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
      {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
      {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
    });

  std::optional<int32_t> motion;
  if (settled) {
    motion = entry_motion_number();
  }
  makeActionSnapshot(/*publish_motor_command=*/!settled, /*profile_velocity=*/settled ? 0.0 : 1.0,
                      /*play_motion_number=*/motion, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::stepPickReady(bool state_changed)
{
  if (state_changed) {
    // (예전엔 여기서 home_q_full로 먼저 홈잉하고 그게 settle될 때까지 기다렸음)
    // 이제는 그 자세로 옮기지 않고 실측 위치를 그대로 IK 시작점(q_)으로 씀 -
    // 단, 매번 그 순간의 motor_position()을 다시 읽으면 그립 실패로 PICK_READY에
    // 재진입했을 때 팔이 뻗어나가 있던(실패한 PICK 시도의) 자세가 그대로
    // "홀드할 위치"가 돼버려서 원래 ready 자세로 다시는 안 돌아오는 문제가 있었음
    // - 이번 pick 사이클에서 PICK_READY에 최초로(SIT 직후) 들어온 순간에만 실측값을
    // latch해두고, 그 뒤로는(재시도 포함) 항상 그 latch값을 씀(pick_ready_q_latched_
    // 필드 주석 참고 - onActivatedImpl의 SIT 진입 시에만 리셋됨).
    if (!pick_ready_q_latched_) {
      pick_ready_latched_q_[0] = motor_position(kTorsoMotorId);
      pick_ready_latched_q_[1] = motor_position(kLeftShoulderPitchMotorId);
      pick_ready_latched_q_[2] = motor_position(kLeftShoulderRollMotorId);
      pick_ready_latched_q_[3] = motor_position(kLeftElbowMotorId);
      pick_ready_q_latched_ = true;
    }
    q_[ik::kTorsoJointIndex] = pick_ready_latched_q_[0];
    q_[ik::kLeftArmIndices[0]] = pick_ready_latched_q_[1];
    q_[ik::kLeftArmIndices[1]] = pick_ready_latched_q_[2];
    q_[ik::kLeftArmIndices[2]] = pick_ready_latched_q_[3];
    damping_ = ik_tuning_.default_damping;
    stuck_streak_ = 0;
    kick_settling_ = false;
    has_captured_target_ = false;
    resetVisionSampleAccumulator();
    torso_bias_commanded_ = false;
    // 이전 시도(예: 그립 실패 재시도) 중에 들어와 아직 안 쓰인 낡은 비전 값이
    // 남아있을 수 있어서, 이번 시도는 항상 새 값으로만 캡처하게 미리 버림.
    discardPendingVision();

    // 지금 위치를 그대로 유지 명령(홀드) - 값 자체가 방금 읽은 실측값이라 실제로는
    // 안 움직임. torso_yaw는 아래 비전 캡처 후 편향값으로 따로 다시 명령함.
    set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
    set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
    set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
    set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);

    // 홀드 명령을 이 틱에 바로 publish함(같은 틱에서 곧장 비전 체크로 넘어가버리면
    // 아직 비전이 없을 때 publish_motor_command=false로 나가버려서 홀드 명령
    // 자체가 유실됨) - 비전 캡처 폴링은 다음 틱부터 시작.
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  // latch된 위치가 최초 진입(SIT 직후)이면 실측값 그대로라 이 체크가 바로
  // 통과하지만, 그립 실패 재시도라면 실측(방금 실패한 PICK 시도로 뻗어나가 있는
  // 자세)과 latch값이 달라서 실제로 그 자리에 도달할 때까지 기다려야 함 - 안
  // 그러면 팔이 아직 돌아오는 중인 자세에서 비전을 캡처해버림.
  if (!allSettled({
      {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
      {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
      {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
      {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
    }))
  {
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  if (!torso_bias_commanded_) {
    // 팔이 latch된 위치에 실제로 도달한 뒤에야(위 settle 체크) 비전 캡처 시도.
    // 아직 못 받았으면 조용히 대기.
    if (!has_captured_target_) {
      // 공(PICK 목표) 캡처 - magnitude bound(0.33)까지 적용.
      has_captured_target_ = tryCapturePendingVision(&target_pos_, /*check_magnitude_bound=*/true);
      if (has_captured_target_) {
        // 비전 y의 실측 오프셋을 캡처 시점에 바로 보정해둠 - 이후 torso 편향
        // 방향 결정은 물론, stepPickIk의 실제 IK 목표(target_pos_)에도
        // 그대로 반영됨.
        target_pos_.y() -= ik_tuning_.vision_y_offset;
        std::printf(
          "[pick_ik_debug] captured target=(%.4f, %.4f, %.4f)\n",
          target_pos_.x(), target_pos_.y(), target_pos_.z());
      } else {
        makeActionSnapshot(/*publish_motor_command=*/false, /*profile_velocity=*/0.0,
                            /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
        return;
      }
    }

    // 비전 캡처가 끝난 뒤에야 torso_yaw를 비전 y 부호로 정한 편향값
    // (+-pick_ready_torso_bias_rad)으로 명령.
    // y가 0에 아주 가까우면(공이 거의 정면) 굳이 허리를 돌릴 필요가 없으므로
    // q_[torso]를 state_changed에서 읽어둔 실측값 그대로 둠(새 명령도 안 보냄) -
    // 그 값 그대로 아래 2단계 settle 확인만 거쳐서 넘어감.
    if (std::fabs(target_pos_.y()) >= 0.02) {
      q_[ik::kTorsoJointIndex] = (target_pos_.y() >= 0.0)
        ? ik_tuning_.pick_ready_torso_bias_rad * 0.0
        : -ik_tuning_.pick_ready_torso_bias_rad;
      set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
    }
    torso_bias_commanded_ = true;
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  // 2단계: torso_yaw가 실제로 그 편향값까지 도달했는지 확인한 뒤에야 PICK으로
  // 전이함 - 그 이후로는 IK가 목표를 향해 필요한 만큼 자유롭게 더 돌림.
  if (!isSettled(kTorsoMotorId, q_[ik::kTorsoJointIndex])) {
    makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                        /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
    return;
  }

  // 이 시점엔 q_(torso + 왼팔 3개)가 다 확정됨 - 이 EE 위치 -> target_pos_(비전
  // 캡처값)를 곧장 damped-IK로 스텝하는 대신, 5차(quintic) 2점 경로로 미리 이어둠
  // (place_trajectory_와 동일 패턴, 정점 없이 시작->목표 직행 - 장애물 회피가
  // 필요 없어서). 점이 2개뿐이라 내부점이 없으므로 estimateVelocities()가 양 끝
  // vel=acc=0인 minimum-jerk 프로파일을 그대로 만들어줌.
  const Eigen::Vector3d approach_start_pos = ik::eePosition(robot_model_, q_);
  pick_approach_trajectory_.clear();
  pick_approach_trajectory_.put_point(
    0.0, approach_start_pos.x(), approach_start_pos.y(), approach_start_pos.z());
  pick_approach_trajectory_.put_point(
    ik_tuning_.pick_approach_duration_sec, target_pos_.x(), target_pos_.y(), target_pos_.z());
  pick_approach_traj_time_ = 0.0;

  advance();  // PICK_READY -> PICK
  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

// stepPickIk/stepPlaceIk가 공유하는 저수준 IK 스텝 실행부 - kick_settling_ 대기 처리,
// ikStep 호출, stuck_streak_/kick 트리거, ik_converged_ 갱신까지 여기서 다 함. 예전엔
// 이 전부가 stepReachableIk(bool, bool) 하나 안에 있었는데, place_tol_ 조정/그리퍼
// 처리가 opening_gripper/use_place_tol 두 불리언 조합별로 갈라지면서 한 함수 안에서
// 읽기 어려워져서 공통부만 이렇게 뺌(stepPickIk/stepPlaceIk 선언부 주석 참고).
std::optional<ik::IkStepResult> ManipulationFSM::runDampedIkTick(double tol)
{
  if (kick_settling_) {
    // 킥 직후: 모터가 실제로 킥된 자세(torso+왼팔)에 도달할 때까지 IK 계산을 멈추고
    // 기다림 - 아직 안 도달했는데 그 q_를 시작점으로 계속 IK를 이어가면 실제 로봇
    // 상태와 어긋난 채로 계산하게 됨(PICK_READY 회전 대기와 동일 이유). 이번 tick엔
    // ikStep을 안 돌렸으므로 nullopt.
    const bool settled = allSettled({
        {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
        {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
        {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
        {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
      });
    if (settled) {
      kick_settling_ = false;
    }
    return std::nullopt;
  }

  // elbowTorsoAvoidanceDq(충돌회피)는 지금 당장은 안 씀 - collision_checker_는 hard
  // 충돌 반려(아래 collision_fn)에는 여전히 쓰이므로 그대로 두고, 여기 secondary_dq는
  // 그리퍼 방향 정렬(top-down 그립) 전용으로만 씀.
  const Eigen::VectorXd secondary_dq =
    ik::eeOrientationAlignmentDq(robot_model_, q_, ik_tuning_.orientation_align_k_pull);

  ik::IkStepParams params;
  params.tol = tol;
  params.alpha = ik_tuning_.alpha;
  params.min_damping = ik_tuning_.min_damping;
  params.max_damping = ik_tuning_.max_damping;
  params.damping_decrease_factor = ik_tuning_.damping_decrease_factor;
  params.damping_increase_factor = ik_tuning_.damping_increase_factor;
  params.joint_weights = active_joint_weights_;
  params.joint_weight_scale = active_joint_weight_scale_;
  params.secondary_dq = secondary_dq;
  params.secondary_gain = ik_tuning_.secondary_gain;
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
  if (stuck_streak_ >= ik_tuning_.stuck_streak_ticks)
  {
    // 고정 후보 목록을 순서대로 대입하던 방식 대신, 목표와 현재 EE의 y 오차 부호를
    // 보고 현재 torso_yaw에서 +/- torso_kick_step_rad만큼 돌림 - 목표가 있는 쪽으로
    // 방향을 잡아서 킥하므로 엉뚱한 방향으로 도는 시행착오를 줄임.
    const Eigen::Vector3d current_ee_pos = ik::eePosition(robot_model_, q_);
    const double error_y = target_pos_.y() - current_ee_pos.y();
    const double kick = q_[ik::kTorsoJointIndex] +
      (error_y < 0.0 ? -ik_tuning_.torso_kick_step_rad : ik_tuning_.torso_kick_step_rad);
    // 팔도 (막혀서 이상하게 꼬여있을 수 있는) 현재 자세 대신 home_q_full로 되돌려서
    // 킥 후 재수렴이 알려진 좋은 자세에서 다시 시작하게 함 - torso_yaw만 kick 값으로
    // 덮어씀.
    q_ = ik_tuning_.home_q_full;
    q_[ik::kTorsoJointIndex] = kick;
    damping_ = ik_tuning_.default_damping;
    stuck_streak_ = 0;
    kick_settling_ = true;
  }

  if (result.converged) {
    ik_converged_ = true;
  }
  return result;
}

void ManipulationFSM::stepPickIk()
{
  if (!ik_converged_) {
    const std::optional<ik::IkStepResult> result = runDampedIkTick(ik_tuning_.tol);
    if (result) {
      // stepPickIk 경로(PICK/COMPLETE_GRIP)는 PLACE의 place_tol_ 같은 유동 완화가
      // 없고 고정 ik_tuning_.tol만 씀 - "거의 다 왔는데 tol을 못 넘어서 멈춘" 상황을
      // 눈으로 확인하려고 매 tick(수렴 여부 무관) 찍음. hard_rejected=0인데
      // damping이 상한 근처에서 안 내려가면 rho가 계속 나빠서(충돌은 아님) 스텝이
      // 사실상 0에 가까워진 정체 상태라는 뜻 - 이땐 stuck_streak_/kick도 안 걸림
      // (그건 hard_rejected 전용이라서, 대화 참고).
      const double error_norm = (target_pos_ - ik::eePosition(robot_model_, q_)).norm();
      std::printf(
        "[pick_stall_debug] converged=%d error_norm=%.4f tol=%.4f damping=%.6f "
        "hard_rejected=%d stuck_streak=%d\n",
        result->converged ? 1 : 0, error_norm, ik_tuning_.tol, damping_,
        result->hard_rejected ? 1 : 0, stuck_streak_);
      if (result->converged) {
        const Eigen::Vector3d converged_ee = ik::eePosition(robot_model_, q_);
        std::printf(
          "[pick_ik_debug] target=(%.4f, %.4f, %.4f) converged_ee=(%.4f, %.4f, %.4f) "
          "diff=(%.4f, %.4f, %.4f) q_torso=%.4f tol=%.4f\n",
          target_pos_.x(), target_pos_.y(), target_pos_.z(),
          converged_ee.x(), converged_ee.y(), converged_ee.z(),
          converged_ee.x() - target_pos_.x(), converged_ee.y() - target_pos_.y(),
          converged_ee.z() - target_pos_.z(), q_[ik::kTorsoJointIndex], ik_tuning_.tol);
      } else if (active_joint_weight_scale_ > 0.0) {
        // 아직 tol 안으로는 안 왔지만 근처(kPickNearTargetDistanceM)까지 왔는데 실측
        // 관절 속도가 전부 정지 수준(kPickStalledJointVelocityRadPerSec 미만)이면,
        // joint_weight_scale 정규화가 남은 오차를 줄이는 걸 막고 있다고 보고 꺼버림.
        // 거리도 IK 내부 시뮬레이션 상태인 q_가 아니라 실측 motor_position()으로 FK를
        // 다시 잡아서 계산함 - q_는 아직 실제 모터가 못 따라간 명령값일 수 있어서
        // (motion_in_flight_ 등), "진짜 지금 EE가 목표에 얼마나 가까운지"는 실측
        // 관절값 기준이어야 정확함.
        Eigen::VectorXd q_real = q_;
        q_real[ik::kTorsoJointIndex] = motor_position(kTorsoMotorId);
        q_real[ik::kLeftArmIndices[0]] = motor_position(kLeftShoulderPitchMotorId);
        q_real[ik::kLeftArmIndices[1]] = motor_position(kLeftShoulderRollMotorId);
        q_real[ik::kLeftArmIndices[2]] = motor_position(kLeftElbowMotorId);
        const double distance_to_target =
          (target_pos_ - ik::eePosition(robot_model_, q_real)).norm();
        const bool near_target = distance_to_target < kPickNearTargetDistanceM;
        const bool joints_barely_moving =
          std::fabs(motor_velocity(kTorsoMotorId)) < kPickStalledJointVelocityRadPerSec &&
          std::fabs(motor_velocity(kLeftShoulderPitchMotorId)) <
            kPickStalledJointVelocityRadPerSec &&
          std::fabs(motor_velocity(kLeftShoulderRollMotorId)) <
            kPickStalledJointVelocityRadPerSec &&
          std::fabs(motor_velocity(kLeftElbowMotorId)) < kPickStalledJointVelocityRadPerSec;
        if (near_target && joints_barely_moving) {
          active_joint_weight_scale_ = 0.0;
        }
      }
    }
  } else {
    gripper_position_ = ik_tuning_.gripper_closed_rad;
  }

  set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
  set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
  set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
  set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);
  set_motor_goal_position(kGripMotorId, gripper_position_);
  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}

void ManipulationFSM::stepPlaceIk(bool opening_gripper)
{
  if (!ik_converged_) {
    const double damping_before = damping_;
    const std::optional<ik::IkStepResult> result = runDampedIkTick(place_tol_);
    if (result) {
      // place_tol_ 유동 조정 - result.converged 하나만 보고 스트릭을 세면, place_tol_이
      // 커서 ikStep이 애초에 dq를 계산도 안 하고("error_norm < tol") 그냥 통과시킨
      // 것까지 "성공"으로 잡혀서 tol을 더 조이려 드는 자기순환이 생김(대화 참고). 대신
      // rho(trust-region ratio)의 결과인 damping 변화로 판단함 - ikStep이 실제로 dq를
      // 계산하고 accept/reject를 결정한 tick인지(real_step = 조기수렴도 아니고 충돌
      // 반려도 아님)부터 걸러내고, 그 안에서 damping이 줄었으면(rho 좋음=진짜 진척)
      // 조이는 쪽, 늘었으면(rho 나쁨, 이때도 q는 그대로임) 또는 충돌로 hard_rejected면
      // 푸는 쪽 스트릭에 반영. rho가 애매(acceptable 구간, damping 불변)하거나
      // result.converged(조기수렴, 근거 없음)면 스트릭을 건드리지 않고 중립으로 둠.
      const bool real_step = !result->converged && !result->hard_rejected;
      if (real_step && damping_ < damping_before) {
        place_tol_fail_streak_ = 0;
        if (++place_tol_success_streak_ >= kPlaceTolAdjustStreakTicks) {
          place_tol_ = std::max(ik_tuning_.tol, place_tol_ - ik_tuning_.place_tol_decrease_step_m);
          place_tol_success_streak_ = 0;
        }
      } else if (result->hard_rejected || (real_step && damping_ > damping_before)) {
        place_tol_success_streak_ = 0;
        if (++place_tol_fail_streak_ >= kPlaceTolAdjustStreakTicks) {
          place_tol_ = std::min(kPlaceTolMaxM, place_tol_ + ik_tuning_.place_tol_increase_step_m);
          place_tol_fail_streak_ = 0;
        }
      }
      std::printf(
        "[place_tol_debug] tol=%.4f converged=%d real_step=%d damping=%.6f->%.6f "
        "success_streak=%d fail_streak=%d\n",
        place_tol_, result->converged ? 1 : 0, real_step ? 1 : 0, damping_before, damping_,
        place_tol_success_streak_, place_tol_fail_streak_);

      // damping이 상한까지 올라갔다는 건 트러스트 리전이 계속 스텝을 반려해서 더는
      // 못 나아간다는 뜻(stepPickIk의 pick_stall_debug와 동일 진단, 대화 참고) - 이
      // 상태로 계속 기다려봐야 더 좁혀지지 않으니 "여기가 한계"로 보고 바로 수렴
      // 처리해서 opening_gripper 쪽 그리퍼 오픈 시퀀스(place_release_error_confirmed_ok_
      // 래치도 같이)로 넘어가게 함 - Cartesian 재검사(real_error) 없이 바로 감.
      if (damping_ >= ik_tuning_.max_damping) {
        std::printf(
          "[place_damping_maxed] damping=%.4f >= max=%.4f - treating as converged\n",
          damping_, ik_tuning_.max_damping);
        ik_converged_ = true;
        place_release_error_confirmed_ok_ = true;
      }

      if (result->converged) {
        const Eigen::Vector3d converged_ee = ik::eePosition(robot_model_, q_);
        std::printf(
          "[place_ik_debug] target=(%.4f, %.4f, %.4f) converged_ee=(%.4f, %.4f, %.4f) "
          "diff=(%.4f, %.4f, %.4f) q_torso=%.4f tol=%.4f\n",
          target_pos_.x(), target_pos_.y(), target_pos_.z(),
          converged_ee.x(), converged_ee.y(), converged_ee.z(),
          converged_ee.x() - target_pos_.x(), converged_ee.y() - target_pos_.y(),
          converged_ee.z() - target_pos_.z(), q_[ik::kTorsoJointIndex], place_tol_);
        // 수렴할 때마다(매 tick 아님) 한 줄 - place_tol_ 유동 조정 폭이 실제 오차와
        // 비교해 타당한지 나중에 로그로 확인하기 위함(위 place_tol_debug는 매 tick
        // 찍혀서 너무 시끄러움).
        std::printf(
          "[place_tol_converged] place_tol_=%.4f err_norm=%.4f success_streak=%d "
          "fail_streak=%d\n",
          place_tol_, (converged_ee - target_pos_).norm(), place_tol_success_streak_,
          place_tol_fail_streak_);
      }
    }
  } else if (opening_gripper) {
    // ik_converged_는 소프트웨어 IK(q_)가 target_pos_에 수렴했다는 뜻일 뿐, 실제
    // 팔(torso+왼팔)이 물리적으로 그 자리에 도달했다는 보장이 아님 - 그 상태에서
    // 바로 그리퍼를 열면 팔이 아직 궤적 끝점으로 이동 중인데 그리퍼부터 벌어지는
    // 문제가 있었음(VIRTUAL_PLACE 전용 - opening_gripper=true인 호출부가 여기뿐).
    // 그리퍼를 열기 전에 실측 피드백으로 진짜 도달을 먼저 확인함. 여기도 다른 곳
    // (advance() 게이트, COMPLETE_GRIP z+lift 대기)과 같은 느슨한 임계값을 재사용함
    // (complete_grip_settle_* - 대화 참고).
    if (!allSettled({
        {kTorsoMotorId, q_[ik::kTorsoJointIndex]},
        {kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]},
        {kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]},
        {kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]},
      }, ik_tuning_.complete_grip_settle_velocity_threshold,
      ik_tuning_.complete_grip_settle_position_tolerance_rad))
    {
      // 아직 실측이 IK 목표에 도달 전 - 그리퍼는 계속 닫힌 채(gripper_position_
      // 유지) 조용히 대기.
    } else {
      // allSettled는 "실측 모터가 q_에 도달했는가"(joint space)만 볼 뿐, 그 q_ 자체가
      // target_pos_ 근처인지(Cartesian)는 안 봄 - place_tol_이 느슨한 채로 ik_converged_가
      // 됐으면 q_가 target_pos_에서 최대 place_tol_(상한 kPlaceTolMaxM=10cm)만큼 떨어진
      // 채로 여기 올 수 있음. place_release_error_confirmed_ok_가 아직 안 래치됐으면
      // 실측 관절값으로 다시 FK를 잡아 독립적으로 재확인해서, 벗어나 있으면 그리퍼를
      // 열지 않고 ik_converged_를 다시 풀어서 재수렴시킴(계속 IK는 풀리게 둠).
      // 한 번이라도 이내로 확인되면 그 뒤로는 재검사 없이 래치 유지 - 아래 그리퍼
      // 오픈 시퀀스(어깨/팔꿈치 오프셋)가 일부러 EE를 target_pos_에서 떨어뜨리는
      // 동작이라, 매 tick 재검사하면 오프셋 적용 후 real_error가 다시 max를 넘어서
      // 이 가드가 q_를 되돌려버리는 충돌이 있었음(대화 참고 - 그리퍼 여는 동안 팔이
      // 흔들리던 원인. place_release_max_error_m/place_release_error_confirmed_ok_
      // 필드 주석 참고).
      if (!place_release_error_confirmed_ok_) {
        Eigen::VectorXd q_real = q_;
        q_real[ik::kTorsoJointIndex] = motor_position(kTorsoMotorId);
        q_real[ik::kLeftArmIndices[0]] = motor_position(kLeftShoulderPitchMotorId);
        q_real[ik::kLeftArmIndices[1]] = motor_position(kLeftShoulderRollMotorId);
        q_real[ik::kLeftArmIndices[2]] = motor_position(kLeftElbowMotorId);
        const double real_error = (target_pos_ - ik::eePosition(robot_model_, q_real)).norm();
        if (real_error > ik_tuning_.place_release_max_error_m) {
          std::printf(
            "[place_release_guard] blocked: real_error=%.4f > max=%.4f - re-solving from real q\n",
            real_error, ik_tuning_.place_release_max_error_m);
          // ik_converged_만 풀고 q_는 그대로 두면, q_ 기준으로는 이미 target_pos_에서
          // place_tol_ 이내(그러니까 애초에 converged=true였던 것)라 다음 tick ikStep이
          // error_norm을 다시 재보자마자 또 즉시 "수렴함"으로 판정해버려서 q_가 전혀
          // 안 바뀐 채로 이 분기에 영원히 되돌아옴(실측으로 확인됨 - real_error가 매
          // tick 똑같은 값으로 안 줄어듦). q_를 실측 위치로 스냅해야 다음 ikStep이
          // "진짜 지금 어디 있는지" 기준으로 다시 계산해서 실제 보정 스텝을 밟음.
          q_ = q_real;
          ik_converged_ = false;
        }
        place_release_error_confirmed_ok_ = real_error <= ik_tuning_.place_release_max_error_m;
      }

      if (place_release_error_confirmed_ok_) {
        // 매 tick 목표를 조금씩(step) 당기지 않고 열림 목표를 한 번에 그대로 명령함 -
        // 실제 이동 속도/궤적은 모터 자체의 profile_velocity가 담당. 완료 판정은
        // 소프트웨어 setpoint가 아니라 실제 모터 피드백(isSettled)으로 함.
        // place_gripper_open_rad 필드 주석 참고 - PICK_READY 진입용 gripper_open_rad와
        // 별개 값(놓을 때 확실히 떨어뜨리려고 더 크게 벌릴 수 있음).
        gripper_position_ = ik_tuning_.place_gripper_open_rad;
        // place_release_shoulder_pitch_commanded_ 필드 주석 참고 - 그리퍼 모터와 연결된
        // 프레임이 놓는 자세에서 그리퍼 아래쪽으로 가버려서 모터만 돌고 실제로는 안
        // 놓아지는 문제 대응. 이 분기에 처음 들어온 tick에만(딱 한 번) 그 순간 실측
        // left_shoulder_pitch + offset을 q_에 바로 대입해서 고정 - 매 tick 다시 읽으면
        // 목표 자체가 계속 움직여서 isSettled가 영영 안 됨(IK는 이미 안 도니 그 뒤로
        // q_가 다른 데서 안 바뀜 - 아래 set_motor_goal_position이 그대로 이 값을 씀).
        if (!place_release_shoulder_pitch_commanded_) {
          q_[ik::kLeftArmIndices[0]] = motor_position(kLeftShoulderPitchMotorId) +
            ik_tuning_.place_release_shoulder_pitch_offset_rad;
          place_release_shoulder_pitch_commanded_ = true;
        }
        // shoulder_pitch만으로 안 빠지는 경우 대비 - 동일 패턴(딱 한 번, 실측값 +
        // offset)으로 elbow_pitch도 같이 돌림(place_release_elbow_offset_rad 필드 주석 참고).
        if (!place_release_elbow_commanded_) {
          q_[ik::kLeftArmIndices[2]] = motor_position(kLeftElbowMotorId) +
            ik_tuning_.place_release_elbow_offset_rad;
          place_release_elbow_commanded_ = true;
        }
        // 여기도 COMPLETE_GRIP z+lift 대기에서 쓰던 것과 같은 느슨한 임계값을 재사용함
        // (complete_grip_settle_* - 이름은 COMPLETE_GRIP 전용처럼 보이지만 실제로는
        // "이 두 군데의 settle 대기 전용 느슨한 값"이라는 뜻으로 같이 씀, 대화 참고).
        if (isSettled(kGripMotorId, ik_tuning_.place_gripper_open_rad,
            ik_tuning_.complete_grip_settle_velocity_threshold,
            ik_tuning_.complete_grip_settle_position_tolerance_rad) &&
          isSettled(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]],
            ik_tuning_.complete_grip_settle_velocity_threshold,
            ik_tuning_.complete_grip_settle_position_tolerance_rad) &&
          isSettled(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]],
            ik_tuning_.complete_grip_settle_velocity_threshold,
            ik_tuning_.complete_grip_settle_position_tolerance_rad))
        {
          advance();  // VIRTUAL_PLACE -> COMPLETE_PLACE
        }
      }
    }
  }
  // [실험] opening_gripper=false로 stepPlaceIk를 부르는 곳(COMPLETE_PLACE의 옛 IK
  // 역재생)이 지금 전부 주석 처리돼있어서, 이 else는 지금 도달 불가능한 죽은 코드임
  // (대화 참고) - 원복하려면 주석을 풀면 됨.
  // else {
  //   gripper_position_ = ik_tuning_.gripper_closed_rad;
  // }

  set_motor_goal_position(kTorsoMotorId, q_[ik::kTorsoJointIndex]);
  set_motor_goal_position(kLeftShoulderPitchMotorId, q_[ik::kLeftArmIndices[0]]);
  set_motor_goal_position(kLeftShoulderRollMotorId, q_[ik::kLeftArmIndices[1]]);
  set_motor_goal_position(kLeftElbowMotorId, q_[ik::kLeftArmIndices[2]]);
  set_motor_goal_position(kGripMotorId, gripper_position_);
  makeActionSnapshot(/*publish_motor_command=*/true, /*profile_velocity=*/1.0,
                      /*play_motion_number=*/std::nullopt, /*deactivate_and_notify=*/false);
}
