#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "ik/collision_model.hpp"
#include "ik/ik_solver.hpp"

// Pick 단계가 전부 끝나야 Place 단계로 넘어감.
enum class Phase
{
  PICK,
  PLACE
};

enum class PickState
{
  SIT,
  PICK_READY,
  PICK,
  COMPLETE_GRIP,
  DONE
};

enum class PlaceState
{
  VIRTUAL_PLACE,
  COMPLETE_PLACE
};

// main_node.cpp/debug_node.cpp가 이 결과 하나만 보고 기계적으로 ROS 액션을 실행함
// (publish/모션 재생/deactivate) - 판단은 전부 ManipulationFSM 안에서 끝내고, 이
// 구조체엔 "무엇을 해야 하는지"와 "지금 상태가 뭔지"(로깅용)만 담음.
struct FsmAction
{
  bool publish_motor_command = false;
  double profile_velocity = 0.0;   // IK 미세 스텝은 0.0(즉시), 홈 텔레포트/회전류는 1.0
  // publish_motor_command가 true일 때만 채워짐(motor_id -> goal position rad) - main_node.cpp가
  // 이 스냅샷만 보고 publish하면 되게 해서 apply_action() 쪽에서 fsm_에 다시 접근할
  // 필요(=락을 또 잡을 필요)가 없게 함("구조체에 관련 정보 몰빵" 요청 반영).
  std::unordered_map<std::string, double> motor_goal_positions;
  std::optional<int32_t> play_motion_number;  // set이면 motion_operator publish
  bool deactivate_and_notify = false;  // deactivate() + activate_cmd(false) publish

  Phase phase = Phase::PICK;
  PickState pick_state = PickState::SIT;
  PlaceState place_state = PlaceState::VIRTUAL_PLACE;
  std::string state_string;  // to_string()과 동일 형식(예: "PICK.PICK_READY")
};

// main_node.cpp의 declare_parameter로 읽은 값들을 한 번에 주입하는 용도.
// 이름/기본값은 config/debug_ik_node.yaml + config/ik_solve_node.yaml + config/main_node.yaml에서
// 이미 검증된 것과 동일하게 맞춤.
struct IkTuning
{
  double default_damping = ik::kDefaultDamping;
  double min_damping = ik::kMinDamping;
  double max_damping = ik::kMaxDamping;
  double damping_decrease_factor = ik::kDampingDecreaseFactor;
  double damping_increase_factor = ik::kDampingIncreaseFactor;
  double tol = 1e-4;
  double alpha = 0.5;
  double trust_region_good_ratio = ik::kTrustRegionGoodRatio;
  double trust_region_acceptable_ratio = ik::kTrustRegionAcceptableRatio;

  // model.nq(8) 크기로 이미 채워서 전달(호출부가 4개 입력값을 torso+왼팔 인덱스에
  // 채우고 나머지를 중립값으로 채우는 패딩을 담당 - debug_ik_node.cpp와 동일 패턴).
  Eigen::VectorXd joint_weights;
  double joint_weight_scale = 0.0;
  double secondary_gain = 0.0;

  double collision_avoidance_threshold_m = 0.03;
  double collision_avoidance_k_pull = 1.0;

  int stuck_streak_ticks = 30;
  std::vector<double> torso_kick_offsets;

  Eigen::VectorXd home_q_full;  // model.nq(8) 크기, PICK_READY 진입 시 복귀할 자세

  double gripper_open_rad = 0.0;
  double gripper_closed_rad = 0.0;
  double gripper_step_rad = 0.0;

  // PICK_READY 전용 "도달 불가능" 대체 로직: 허리+왼쪽 어깨 롤을 이 각도로 고정
  // 회전시키고, 위치/속도 둘 다 임계값 안이면 도달로 봄.
  double torso_target_rad = 1.5;
  double left_shoulder_roll_target_rad = 1.0;
  double settle_velocity_threshold = 0.3;  // CurrentMotorStatus.velocity는 raw uint8 단위
  double settle_position_tolerance_rad = 0.05;

  // 그립 모터(6번) 정지 판정(main_node.yaml의 grip.* 그대로).
  double grip_moving_velocity_threshold = 0.3;
  double grip_near_zero_position_threshold = 0.2;
  double grip_wiggle_kick_position_rad = 1.0;

  // COMPLETE_PLACE 홈잉 정지 판정(main_node.yaml의 arm.settled_velocity_threshold 그대로).
  double arm_settled_velocity_threshold = 0.3;
};

class ManipulationFSM
{
public:
  // ik::RobotModel이 기본 생성자가 없어서(URDF를 반드시 로드해야 함) 이 클래스도
  // urdf_path를 반드시 받아야 함(기존 기본 생성자는 제거).
  explicit ManipulationFSM(const std::string & urdf_path, const std::string & ee_frame = "ee_link_1");

  // collision_checker_가 robot_model_을 참조(RobotModel&)로 들고 있어서(ik/collision_model.hpp
  // 참고) 복사/이동하면 그 참조가 옛 메모리를 가리키게 됨(dangling) - 아예 막아서
  // 실수로 값 전달/return-by-value 하면 컴파일 에러가 나게 함(main_node.cpp/debug_node.cpp는
  // std::unique_ptr<ManipulationFSM>로 들고 있어서 이 제약과 무관함).
  ManipulationFSM(const ManipulationFSM &) = delete;
  ManipulationFSM & operator=(const ManipulationFSM &) = delete;
  ManipulationFSM(ManipulationFSM &&) = delete;
  ManipulationFSM & operator=(ManipulationFSM &&) = delete;

  // 생성자 직후 1회 호출 - yaml에서 읽은 튜닝값 일괄 주입.
  void configure_ik(const IkTuning & tuning);

  // 현재 상태의 다음 단계로 전이.
  // Pick의 마지막 상태(DONE)에서 advance()하면 Place 단계로 넘어가
  // VIRTUAL_PLACE부터 시작함. Place의 마지막 상태(COMPLETE_PLACE)에서는
  // 더 전이하지 않고 false를 반환함(전체 시퀀스 종료).
  bool advance();

  void reset();

  Phase phase() const;
  PickState pick_state() const;
  PlaceState place_state() const;

  bool is_sequence_complete() const;

  // motor_status 콜백에서 모터 하나의 현재 위치/속도를 갱신. motor_id는 "1", "10"
  // 처럼 모터 번호를 문자열로 넣음. 판단은 안 하고 그대로 저장만 함(판단은
  // on_motor_feedback()이 함) - main_node.cpp/debug_node.cpp를 최대한 얇게 유지하기 위함.
  void set_motor_position(const std::string & motor_id, double position);
  void set_motor_velocity(const std::string & motor_id, double velocity);

  // motor_id로 등록해둔 최근 위치(rad)를 조회. 값을 한 번도 못 받은 id면 0.0.
  double motor_position(const std::string & motor_id) const;

  // IK 등에서 계산한 모터 목표 위치(rad)를 등록/조회.
  void set_motor_goal_position(const std::string & motor_id, double goal_position);
  double motor_goal_position(const std::string & motor_id) const;

  // /master2mani(또는 /vision) 콜백마다 그대로 호출 - 최신 원시값만 저장함(판단 없음).
  // PICK_READY/VIRTUAL_PLACE가 허리+어깨 회전을 시작하기 "직전"에 이 값을 딱 한 번
  // 캡처해서 그 뒤로는(같은 시도 안에서는) 재조회하지 않음 - solve_tick() 내부에서 처리.
  void update_vision(double x, double y, double z);

  // 현재 상태에 진입할 때 재생해야 하는 모션 번호(motion_operator의 motion_num).
  // SIT=73, COMPLETE_GRIP=74. 재생할 모션이 없는 상태면 std::nullopt.
  std::optional<int32_t> entry_motion_number() const;

  std::string to_string() const;

  // --- 아래 4개가 main_node.cpp/debug_node.cpp가 실제로 호출하는 진입점(전부
  // xxxImpl()로 action_을 갱신한 뒤 finalize()로 마무리하는 얇은 wrapper) ---

  // on_activate() 직후 1회 호출(SIT 진입 모션 재생 여부 판단).
  FsmAction on_activated() {onActivatedImpl(); return finalize();}

  // motor_status 콜백마다(위 setter들을 채운 뒤) 1회 호출 - 그립 정지 판정/그립
  // 위글 시퀀스/COMPLETE_PLACE 직접 홈잉을 전부 여기서 처리.
  FsmAction on_motor_feedback() {onMotorFeedbackImpl(); return finalize();}

  // motion_end 콜백마다 호출 - SIT 종료(-> PICK_READY 전이 + 다리/반대팔 고정),
  // COMPLETE_GRIP 모션 종료(-> 그립 위글 시퀀스 시작) 처리. motion_num은 로직에서
  // 안 쓰여서(phase/pick_state 기준으로만 판단) 받지 않음.
  FsmAction on_motion_end(bool ended) {onMotionEndImpl(ended); return finalize();}

  // 전용 스레드 타이머에서 매 주기 1회 호출 - PICK_READY(홈복귀->비전 1회 캡처->
  // 허리+어깨 회전), PICK/VIRTUAL_PLACE(도달 가능 IK + 그리퍼) 진행.
  FsmAction solve_tick() {solveTickImpl(); return finalize();}

private:
  static std::string pick_state_name(PickState s);
  static std::string place_state_name(PlaceState s);

  // 상태(phase/pick_state/place_state/state_string)는 항상 지금 값으로 새로 채우고,
  // 나머지 필드는 인자로 받은 값 그대로 세팅함 - 매번 새 FsmAction을 만드는 대신
  // 재사용되는 action_ 멤버를 갱신하고 그 참조를 돌려줌. phase/pick_state 등이
  // advance() 호출로 바뀔 때마다 다시 불러서 최신 상태로 갱신하면 됨. 기본값을
  // 일부러 안 둠 - 호출부마다 4개를 전부 명시하게 강제해서(/*name=*/ 주석과 함께)
  // "여기는 3개만 넘겼네" 식으로 헷갈리는 일이 없게 함.
  FsmAction & makeActionSnapshot(bool publish_motor_command, double profile_velocity,
                                  std::optional<int32_t> play_motion_number,
                                  bool deactivate_and_notify);
  // publish_motor_command가 true면 motor_goal_positions_ 스냅샷을 채워서 action_을 반환.
  // 4개 공개 진입점의 공통 출구.
  FsmAction finalize();
  void onActivatedImpl();
  void onMotorFeedbackImpl();
  void onMotionEndImpl(bool ended);
  void solveTickImpl();

  bool checkGripMotorStopped(double motor6_velocity);
  // 모터 하나가 목표 위치(rad)에 도달했는지(위치+속도 둘 다 임계값 안) 판정하는
  // 유일한 통로 - 회전 settle/COMPLETE_PLACE 홈잉 settle이 전부 이걸 통해서만 판정함.
  bool isSettled(const std::string & motor_id, double target_rad) const;
  // motor_id -> 목표 위치(rad) 목록을 받아서 전부 isSettled()를 만족하는지 확인.
  bool allSettled(const std::vector<std::pair<std::string, double>> & targets) const;

  // solve_tick()이 위임하는 phase/state별 처리. state_changed는 "지난 solve_tick()
  // 호출 이후로 phase/pick_state/place_state가 바뀌었는지"(자체 감지, 별도 신호 불필요).
  void stepPickReady(bool state_changed);
  void stepReachableIk(bool opening_gripper);
  void stepCompletePlaceHoming(bool state_changed);

  Phase phase_;
  PickState pick_state_;
  PlaceState place_state_;

  std::unordered_map<std::string, double> motor_positions_;
  std::unordered_map<std::string, double> motor_velocities_;
  std::unordered_map<std::string, double> motor_goal_positions_;
  bool grip_motor_was_moving_ = false;

  bool has_pending_vision_ = false;
  double vision_x_ = 0.0;
  double vision_y_ = 0.0;
  double vision_z_ = 0.0;

  // --- IK 컨트롤러 상태(구 ik_solve_node.py의 인스턴스 변수들) ---
  // 선언 순서 중요: collision_checker_ 생성자가 robot_model_ 참조를 받으므로
  // robot_model_이 먼저 와야 함(collision_model.hpp의 CollisionChecker와 동일한 이유).
  ik::RobotModel robot_model_;
  ik::CollisionChecker collision_checker_;
  IkTuning ik_tuning_;

  Eigen::VectorXd q_;
  double damping_ = ik::kDefaultDamping;
  Eigen::Vector3d target_pos_ = Eigen::Vector3d::Zero();
  bool has_captured_target_ = false;
  bool rotation_commanded_ = false;
  bool ik_converged_ = false;
  int stuck_streak_ = 0;
  size_t torso_kick_index_ = 0;
  bool kick_settling_ = false;
  double gripper_position_ = 0.0;

  Phase prev_solve_phase_;
  PickState prev_solve_pick_state_;
  PlaceState prev_solve_place_state_;

  // 그립 위글 시퀀스(COMPLETE_GRIP 모션 종료 후 2틱에 걸쳐 kick->restore).
  bool pending_grip_wiggle_ = false;
  bool grip_wiggle_kicked_ = false;
  double cached_grip_position_ = 0.0;

  // makeActionSnapshot()/finalize()가 매번 새로 만들지 않고 계속 재사용하는 액션.
  FsmAction action_;
};
