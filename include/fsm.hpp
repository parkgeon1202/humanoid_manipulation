#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "ik/collision_model.hpp"
#include "ik/ik_solver.hpp"
#include "ik/trajectory_generator.hpp"

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

// main_node.cpp가 이 결과 하나만 보고 기계적으로 ROS 액션을 실행함
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
  bool deactivate_and_notify = false;  // activate_end(true) publish + deactivate()
  // true면 카메라를 레벨(tilt=0)로 되돌리는 PanTilt를 한 번 publish(pan은 안 건드림) -
  // 모션 75(DONE 재활성화 진입 모션) 종료 시점(onMotionEndImpl)에만 세팅됨.
  bool reset_pan_tilt_level = false;

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
  // PLACE(VIRTUAL_PLACE) 전용 joint_weights - PICK과 반대로 torso_yaw를 더 아끼고
  // (허리를 덜 쓰고) 팔 위주로 풀게 하려는 용도. stepPickIk/stepPlaceIk는 phase에 맞는
  // 쪽을 state_changed 시점에 active_joint_weights_로 스냅샷해서 씀(solveTickImpl 참고).
  Eigen::VectorXd place_joint_weights;
  // COMPLETE_PLACE(역재생 홈잉) 전용 joint_weights - stepCompletePlaceHoming
  // state_changed에서 active_joint_weights_로 스냅샷됨. torso_yaw를 크게 아껴서
  // (기본 100) 돌아올 때 허리 대신 팔 위주로 풀게 함.
  Eigen::VectorXd complete_place_joint_weights;
  double joint_weight_scale = 0.0;
  // EE 위치 오차 항(가중치 1)과 견줄 만큼 secondary_dq(그리퍼 방향 정렬)도 중요하게
  // 다루려고 기존 충돌회피용 기본값(0.01)보다 훨씬 크게 잡음(실기에서 더 튜닝 필요).
  double secondary_gain = 1.0;
  double orientation_align_k_pull = 1.0;

  double collision_avoidance_threshold_m = 0.03;
  double collision_avoidance_k_pull = 1.0;

  int stuck_streak_ticks = 30;
  // 정체 킥 때 팔꿈치를 펴둔 채로(0.0) 두면 다리/공을 치는 경우가 있어서, torso_yaw
  // 킥과 함께 팔꿈치도 이 각도(rad)만큼 굽혀서 리셋함.
  double elbow_kick_rad = 0.0;
  // torso_yaw 킥 크기(rad). 정체 시 목표와 현재 EE의 y 오차 부호를 보고 현재
  // torso_yaw에서 이 값만큼 +/- 방향으로 돌림(stepPlaceIk 참고).
  double torso_kick_step_rad = 0.5;
  // 위와 동일한 이유로 어깨 pitch도 0.0 대신 이 각도(rad)로 리셋함.
  double left_shoulder_pitch_kick_rad = -2.7;

  Eigen::VectorXd home_q_full;  // model.nq(8) 크기, PICK_READY 진입 시 복귀할 자세

  // COMPLETE_GRIP 진입 직후(stepCompleteGripHoming) 고정 프리셋 관절각으로 스냅하는
  // 대신, 방금 집은 자세의 EE 위치(FK)에서 z를 이만큼(m) 들어올린 지점을 새 IK
  // 목표로 풀어서 자세를 잡음.
  double complete_grip_lift_z_m = 0.1;

  double gripper_open_rad = 0.0;
  double gripper_closed_rad = 0.0;
  // VIRTUAL_PLACE에서 물건을 놓을 때 벌리는 목표(stepPlaceIk 참고) - PICK_READY
  // 진입 시 미리 벌려두는 gripper_open_rad와 별개 값. 그냥 집을 때보다 확실히
  // 떨어뜨리려면 더 크게 벌려야 할 수 있어서 따로 뺌.
  double place_gripper_open_rad = 0.0;
  // VIRTUAL_PLACE에서 그리퍼를 벌릴 때 left_shoulder_pitch도 그 순간 실측값 기준으로
  // 이만큼(rad) 같이 돌림(stepPlaceIk 참고) - 그리퍼 모터와 연결된 프레임이
  // 놓는 자세에서 그리퍼 아래쪽으로 가버려서 모터만 돌아가고 실제로는 안 놓아지는
  // 문제 대응(어깨를 더 돌려서 프레임 위치를 바꿔줌).
  double place_release_shoulder_pitch_offset_rad = 0.0;
  // 위와 동일 시점(그리퍼 벌릴 때 한 번)에 elbow_pitch도 그 순간 실측값 기준으로
  // 이만큼(rad) 같이 돌림 - shoulder_pitch만으로 부족할 때 팔꿈치까지 보태서
  // 프레임을 확실히 그리퍼 아래쪽에서 빼냄.
  double place_release_elbow_offset_rad = 0.0;

  // 정체 킥 때 어깨 롤을 이 각도로 리셋함(elbow_kick_rad/left_shoulder_pitch_kick_rad와
  // 짝, kick 전용).
  double left_shoulder_roll_target_rad = 0.3;

  // PICK_READY 진입 시 torso_yaw를 고정 프리셋(home_q_full) 대신 비전 y 부호로
  // 미리 그쪽으로 기울여두는 크기(rad) - stepPickReady 참고. y가 오른쪽(음수)이면
  // -이 값, 왼쪽(양수)이면 +이 값으로 씀.
  double pick_ready_torso_bias_rad = 1.0;

  double settle_velocity_threshold = 0.3;  // CurrentMotorStatus.velocity는 rad/s 단위
  double settle_position_tolerance_rad = 0.05;

  // COMPLETE_GRIP z+lift 목표 전용(stepCompleteGripHoming 참고) - 위 공용 settle
  // 임계값(다른 곳의 허리/어깨 회전 대기 등에도 쓰임)을 그대로 쓰면 이 구간만 실측
  // settle까지 15~20초씩 걸리는 문제가 있어서(대화 참고) 이 대기 전용으로 더 느슨한
  // 값을 따로 둠 - isSettled/allSettled의 override 인자로 넘겨서 다른 곳엔 영향 없음.
  double complete_grip_settle_velocity_threshold = 0.6;
  double complete_grip_settle_position_tolerance_rad = 0.6;

  // 그립 모터(6번) 정지 판정(main_node.yaml의 grip.* 그대로).
  double grip_moving_velocity_threshold = 0.5;
  double grip_near_zero_position_threshold = 0.1;
  double grip_wiggle_kick_position_rad = 1.0;
  // velocity가 threshold 밑으로 이 tick 수만큼 연속으로 유지돼야 "진짜 멈췄다"로
  // 확정함(1틱만 보고 확정하면 정지 마찰/토크 리플 같은 순간적인 속도 저하를
  // 오탐해서 훨씬 덜 오므린 채로 그립 완료 처리되는 문제가 있었음).
  int grip_stopped_streak_ticks = 10;

  // COMPLETE_PLACE 홈잉 정지 판정(main_node.yaml의 arm.settled_velocity_threshold 그대로).
  double arm_settled_velocity_threshold = 0.3;

  // 그립 위글 복원(다시 오므리기) settle 대기가 이 틱 수를 넘으면 isSettled 결과와
  // 무관하게 settle된 것으로 간주하고 넘어감 - 그리퍼가 물체를 물어서 목표
  // 위치(gripper_closed_rad)까지 완전히 안 닫히는 경우 여기서 무한 대기하는 걸
  // 막는 타임아웃.
  int grip_wiggle_restore_timeout_ticks = 200;

  // 그립 위글 kick(그리퍼를 잠깐 벌리는) 단계가 실제로 그 위치까지 도달했는지 확인할
  // 전용 settle 임계값 - 공용 ik.torso_settle_*(현재 0.3rad로 느슨하게 잡혀있음)을
  // 그대로 쓰면 실제로는 별로 안 벌어졌는데도 settle로 오판할 여지가 있어서, 이 확인만
  // 더 타이트하게 따로 둠.
  double grip_wiggle_kick_settle_velocity_threshold = 0.2;
  double grip_wiggle_kick_settle_position_tolerance_rad = 0.1;

  // VIRTUAL_PLACE의 place_trajectory_ 재생 시간을 매 틱 얼마나 진행시킬지(ik.step_period_sec와
  // 동일 값을 넣어서 씀 - solve_tick 호출 주기와 궤적 재생 속도를 맞추기 위함).
  double place_traj_dt_sec = 0.02;

  // VIRTUAL_PLACE 팔 궤적: 시작(현재 EE) -> 정점(장애물 회피, 고정 좌표) -> 목표(비전
  // 캡처값)를 5차 다항식 경유점 3개로 이음. 정점은 duration/3 지점, 목표는 duration
  // 지점 - z는 시작/정점/목표 세 점 모두 같은 vz를 줘서 등속 직선이 되게 함(경계조건이
  // 일치하면 quintic이 정확히 직선으로 퇴화하는 성질 이용 - Trajectory1D 참고).
  double place_apex_x = 0.1;
  double place_apex_y = 0.3;
  double place_traj_duration_sec = 3.0;  // 시작->정점->목표 전체 소요 시간(초)

  // VIRTUAL_PLACE place_tol_ 유동 조정 폭(m, runDampedIkTick/stepPlaceIk 참고) - 예전엔
  // 늘고 주는 방향을 비대칭(늘리는 쪽을 더 크게)으로 뒀었는데, 그건 "그냥 result.converged
  // 여부"로 스트릭을 셀 때(place_tol_이 커서 생긴 가짜 수렴까지 성공으로 잡히던 문제)의
  // 임시방편이었음. 지금은 rho/damping으로 "진짜 스텝을 밟았는지"부터 걸러내므로 비대칭을
  // 유지할 근거가 없어져서 같은 값으로 맞춤.
  double place_tol_increase_step_m = 0.01;
  double place_tol_decrease_step_m = 0.01;
  // 그리퍼를 열기 전 마지막 안전판(stepPlaceIk 참고) - allSettled()는 실측 모터가
  // q_에 도달했는지(joint space)만 보고 q_ 자체가 target_pos_ 근처인지(Cartesian)는
  // 안 보므로, 실측 관절값으로 다시 FK를 잡아 target_pos_와의 거리가 이 값을 넘으면
  // 그리퍼를 열지 않고 ik_converged_를 다시 false로 풀어서 재수렴을 시킴. place_tol_의
  // 상한(kPlaceTolMaxM, 10cm)보다 훨씬 타이트하게 잡아야 의미가 있음.
  double place_release_max_error_m = 0.02;

  // 비전 y값의 실측 오프셋(m) - PICK_READY의 torso_yaw 편향 방향 결정 시
  // target_pos_.y()에서 이 값을 뺀 보정값(corrected_y)을 씀(stepPickReady 참고).
  double vision_y_offset = 0.0328;

  // PICK 접근 궤적(pick_approach_trajectory_): PICK_READY -> PICK 전이 직전(q_ 4개가
  // 다 확정된 시점)의 EE 위치 -> target_pos_ 2점만으로 잇는 5차(quintic) 경로 -
  // place_trajectory_와 달리 회피할 장애물이 없어 정점 없이 직행. 양 끝 vel=acc=0
  // (TrajectoryGenerator::put_point 기본값)이라 minimum-jerk 프로파일로 부드럽게
  // 출발/도착함(진짜 3차는 아니고 Trajectory1D가 항상 쓰는 quintic 그대로 재사용).
  double pick_approach_duration_sec = 1.0;
};

class ManipulationFSM
{
public:
  // ik::RobotModel이 기본 생성자가 없어서(URDF를 반드시 로드해야 함) 이 클래스도
  // urdf_path를 반드시 받아야 함(기존 기본 생성자는 제거).
  explicit ManipulationFSM(const std::string & urdf_path, const std::string & ee_frame = "ee_link_1");

  // collision_checker_가 robot_model_을 참조(RobotModel&)로 들고 있어서(ik/collision_model.hpp
  // 참고) 복사/이동하면 그 참조가 옛 메모리를 가리키게 됨(dangling) - 아예 막아서
  // 실수로 값 전달/return-by-value 하면 컴파일 에러가 나게 함(main_node.cpp는
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

  // RViz 시각화 전용(main_node.cpp가 매 tick 그대로 publish) - solve_tick 스레드에서만
  // 쓰여서(다른 로직 전용 상태와 동일 이유) 별도 락 없음.
  // 현재 IK가 쫓고 있는 목표점(target_pos_) - 아직 캡처 전이면 Zero().
  Eigen::Vector3d target_pos() const;
  // 지금 활성 상태(PICK.PICK면 pick_approach_trajectory_, PLACE.VIRTUAL_PLACE면
  // place_trajectory_)의 경로를 [0, duration] 구간에서 균등 샘플링해서 돌려줌 - 그
  // 외 상태면 활성 궤적이 없으므로 빈 벡터. num_samples는 양 끝점 포함 개수(2 이상).
  std::vector<Eigen::Vector3d> sample_active_trajectory(int num_samples = 30) const;

  // motor_status 콜백에서 모터 하나의 현재 위치/속도를 갱신. motor_id는 "1", "10"
  // 처럼 모터 번호를 문자열로 넣음. 판단은 안 하고 그대로 저장만 함(판단은
  // on_motor_feedback()이 함) - main_node.cpp를 최대한 얇게 유지하기 위함.
  // 메시지 콜백 스레드(쓰기)와 solve_tick 스레드(읽기)가 동시에 접근하므로 함수
  // 내부에서 data_mutex_로 짧게 보호함(호출부는 락을 신경 쓸 필요 없음).
  void set_motor_position(const std::string & motor_id, double position);
  void set_motor_velocity(const std::string & motor_id, double velocity);

  // motor_id로 등록해둔 최근 위치/속도를 조회. 값을 한 번도 못 받은 id면 0.0.
  double motor_position(const std::string & motor_id) const;
  double motor_velocity(const std::string & motor_id) const;

  // 목(pan/tilt) 모터 전용 - kMotorIdOrder에 없어서(tilt는 이 시스템에서 모터로
  // 제어 안 함) motor_positions_/motor_velocities_ 맵에 안 들어가고, main_node.cpp가
  // CurrentMotorStatus.position/velocity의 마지막 원소(tilt 실측값)를 여기로 따로
  // 넣어줌. VIRTUAL_PLACE 진입 시 카메라가 레벨(tilt=0)로 실제로 돌아왔는지 확인하는
  // 용도(isTiltSettled() 참고).
  void set_tilt_position(double position);
  void set_tilt_velocity(double velocity);

  // IK 등에서 계산한 모터 목표 위치(rad)를 등록/조회. solve_tick 스레드에서만
  // 쓰여서(로직 메서드 전용) 별도 락 없음.
  void set_motor_goal_position(const std::string & motor_id, double goal_position);
  double motor_goal_position(const std::string & motor_id) const;

  // /master2mani(또는 /vision) 콜백마다 그대로 호출 - 최신 원시값만 저장함(판단 없음).
  // 메시지 콜백 스레드(쓰기)와 solve_tick 스레드(읽기)가 동시에 접근하므로 함수
  // 내부에서 data_mutex_로 짧게 보호함. PICK_READY/VIRTUAL_PLACE가 허리+어깨 회전을
  // 시작하기 "직전"에 이 값을 딱 한 번 캡처해서 그 뒤로는(같은 시도 안에서는)
  // 재조회하지 않음(tryCapturePendingVision() 참고).
  void update_vision(double x, double y, double z);

  // motion_end 콜백에서 그대로 호출 - 로직 없이 이벤트만 기록함(실제 처리는
  // solve_tick 스레드가 take_pending_motion_end()로 소비할 때 onMotionEndImpl()을
  // 통해 일어남 - 로직 메서드는 전부 solve_tick 스레드 하나에서만 돌게 하기 위함).
  void notify_motion_end(bool ended);

  // solve_tick 스레드 전용: 대기 중인 motion_end 이벤트가 있으면 꺼내서(그 자리에서
  // 지움) 돌려주고, 없으면 nullopt.
  std::optional<bool> take_pending_motion_end();

  // 현재 상태에 진입할 때 재생해야 하는 모션 번호(motion_operator의 motion_num).
  // SIT=73, COMPLETE_GRIP=74, DONE(picking 끝나고 place용으로 재활성화됐을 때)=75,
  // COMPLETE_PLACE(역재생 IK가 시작점까지 수렴 + settle된 뒤 전신을 최종 자세로
  // 되돌림)=76. 재생할 모션이 없는 상태면 std::nullopt.
  std::optional<int32_t> entry_motion_number() const;

  std::string to_string() const;

  // --- 아래 4개가 main_node.cpp가 실제로 호출하는 진입점(전부
  // xxxImpl()로 action_을 갱신한 뒤 finalize()로 마무리하는 얇은 wrapper) ---

  // on_activate() 직후 1회 호출(SIT 진입 모션, 또는 DONE 상태에서 place용으로
  // 재활성화된 경우 그 진입 모션의 재생 여부 판단).
  FsmAction on_activated() {onActivatedImpl(); return finalize();}

  // motor_status 콜백마다(위 setter들을 채운 뒤) 1회 호출 - 그립 정지 판정/그립
  // 위글 시퀀스/COMPLETE_PLACE 직접 홈잉을 전부 여기서 처리.
  FsmAction on_motor_feedback() {onMotorFeedbackImpl(); return finalize();}

  // motion_end 콜백마다 호출 - SIT 종료(-> PICK_READY 전이 + 다리/반대팔 고정),
  // DONE 진입 모션 종료(-> PLACE.VIRTUAL_PLACE 전이 + 다리/반대팔 고정), COMPLETE_GRIP
  // 모션 종료(-> 그립 위글 시퀀스 시작) 처리. motion_num은 로직에서 안 쓰여서
  // (phase/pick_state 기준으로만 판단) 받지 않음.
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

  // has_pending_vision_/vision_x_/y_/z_ 읽기+캡처+클리어를 data_mutex_ 안에서
  // 원자적으로 처리 - update_vision()이 다른 스레드에서 이 필드들을 쓸 수 있어서.
  // 논블로킹: pending 값이 있으면 하나만 소비해 vision_sample_sum_/count_(멤버,
  // tick 간 유지)에 누적하고, 아직 kVisionSampleTarget(10)개를 못 모았으면 false를
  // 돌려줌 - 호출부(solve_tick)는 다음 tick에 새 pending 값이 들어오면 다시 시도하는
  // 기존 재시도 패턴을 그대로 씀. 다 모으면 평균을 *out에 채우고 누적 상태를 리셋한
  // 뒤 true를 돌려줌(첫 pending 값조차 없으면 즉시 false). x/y/z 셋 다 0.0 또는 셋
  // 다 kVisionInvalidSentinel(-0.999)인 통짜 센티널(탐지 실패)은 항상 버림.
  // check_magnitude_bound가 true면(공/PICK_READY 전용) 거기에 더해 |x|,|y|,|z|가
  // 셋 다 kVisionMagnitudeBound(0.33)를 넘는 표본도 버림 - 골대/VIRTUAL_PLACE
  // 캡처는 false로 호출해서 이 범위 제한 없이 센티널 필터만 적용함.
  bool tryCapturePendingVision(Eigen::Vector3d * out, bool check_magnitude_bound);

  // vision_sample_sum_/count_를 0으로 리셋(data_mutex_로 보호) - 새 캡처 시도를
  // 시작하기 전, 이전 시도에서 일부만 모으고 중단된 누적값을 이어받지 않게 함.
  void resetVisionSampleAccumulator();

  // PICK_READY/VIRTUAL_PLACE state_changed 시점처럼, 이전 시도 중에 받아둔 낡은
  // 비전 값을 이번 시도의 캡처값으로 쓰지 않게 미리 버림(data_mutex_로 보호).
  void discardPendingVision();

  bool checkGripMotorStopped(double motor6_velocity);
  // 모터 하나가 목표 위치(rad)에 도달했는지(위치+속도 둘 다 임계값 안) 판정하는
  // 유일한 통로 - 회전 settle/COMPLETE_PLACE 홈잉 settle이 전부 이걸 통해서만 판정함.
  // velocity_threshold_override/position_tolerance_override: 생략하면(nullopt)
  // ik_tuning_.settle_velocity_threshold/settle_position_tolerance_rad를 그대로 씀 -
  // 특정 대기 구간만 다른 임계값을 쓰고 싶을 때만(예: stepCompleteGripHoming의
  // complete_grip_settle_*) 채워서 넘김.
  bool isSettled(const std::string & motor_id, double target_rad,
                 std::optional<double> velocity_threshold_override = std::nullopt,
                 std::optional<double> position_tolerance_override = std::nullopt) const;
  // motor_id -> 목표 위치(rad) 목록을 받아서 전부 isSettled()를 만족하는지 확인.
  bool allSettled(const std::vector<std::pair<std::string, double>> & targets,
                   std::optional<double> velocity_threshold_override = std::nullopt,
                   std::optional<double> position_tolerance_override = std::nullopt) const;
  // isSettled()와 동일한 판정(속도+위치 임계값)을 tilt_position_/tilt_velocity_에
  // 대해 목표 0.0(레벨)로 확인 - motor_positions_ 맵을 안 쓰는 tilt 전용이라 별도로
  // 뺌. 모션 재생(motion_in_flight_)과 무관하게(목 모터는 motion_operator가 아니라
  // /RealsensePanTilt로만 구동됨) 항상 판정함.
  bool isTiltSettled() const;

  // solve_tick()이 위임하는 phase/state별 처리. state_changed는 "지난 solve_tick()
  // 호출 이후로 phase/pick_state/place_state가 바뀌었는지"(자체 감지, 별도 신호 불필요).
  void stepPickReady(bool state_changed);
  // PICK(공 접근)/COMPLETE_GRIP(들어올린 자세로 복귀) 전용 - 둘 다 tol은
  // ik_tuning_.tol 고정, 그리퍼는 항상 닫힌 채로 유지함. 예전엔 이 둘과 아래
  // stepPlaceIk를 stepReachableIk(bool, bool) 하나로 겸용했는데, place_tol_
  // 조정/그리퍼 오픈 로직이 붙으면서 두 불리언 조합별로 갈라지는 분기가 너무
  // 읽기 어려워져서 용도별로 나눔 - 공통 IK 스텝 실행부만 runDampedIkTick()으로 공유.
  void stepPickIk();
  // VIRTUAL_PLACE(place_trajectory_ 순방향 추적, opening_gripper=true - 도착하면
  // 그리퍼를 열고 advance())/COMPLETE_PLACE(같은 궤적을 역재생, opening_gripper=false -
  // 그리퍼는 안 건드림) 공용 - 둘 다 유동 place_tol_을 씀.
  void stepPlaceIk(bool opening_gripper);
  // stepPickIk/stepPlaceIk가 공유하는 저수준 IK 스텝 실행부 - kick_settling_ 대기,
  // ikStep 호출 + stuck_streak_/kick 트리거까지 처리하고 q_/damping_/ik_converged_까지
  // 갱신함(converged면 ik_converged_=true). kick_settling_ 대기 중이라 이번 tick엔
  // ikStep 자체를 안 돌렸으면 std::nullopt를 돌려줌 - 호출부(stepPlaceIk의 place_tol_
  // 조정 등)가 "이번 tick에 실제로 판단할 근거가 있었는지"를 구분해야 해서.
  std::optional<ik::IkStepResult> runDampedIkTick(double tol);
  void stepCompleteGripHoming(bool state_changed);
  void stepDoneHoming(bool state_changed);
  void stepCompletePlaceHoming(bool state_changed);

  Phase phase_;
  PickState pick_state_;
  PlaceState place_state_;

  // 메시지 콜백 스레드(motor_status/master2mani/motion_end)와 solve_tick 스레드가
  // 공유하는 원시 데이터(motor_positions_/motor_velocities_/vision*/pending_motion_end_)만
  // 보호함 - motor_goal_positions_ 등 로직 전용 상태는 solve_tick 스레드 하나에서만
  // 쓰여서 이 락이 필요 없음.
  mutable std::mutex data_mutex_;

  std::unordered_map<std::string, double> motor_positions_;
  std::unordered_map<std::string, double> motor_velocities_;
  std::unordered_map<std::string, double> motor_goal_positions_;
  bool grip_motor_was_moving_ = false;
  int grip_stopped_streak_ = 0;

  // set_tilt_position()/set_tilt_velocity() 참고 - 목(tilt) 모터 실측값 전용.
  double tilt_position_ = 0.0;
  double tilt_velocity_ = 0.0;

  bool has_pending_vision_ = false;
  double vision_x_ = 0.0;
  double vision_y_ = 0.0;
  double vision_z_ = 0.0;

  std::optional<bool> pending_motion_end_;

  // --- IK 컨트롤러 상태(구 ik_solve_node.py의 인스턴스 변수들) ---
  // 선언 순서 중요: collision_checker_ 생성자가 robot_model_ 참조를 받으므로
  // robot_model_이 먼저 와야 함(collision_model.hpp의 CollisionChecker와 동일한 이유).
  ik::RobotModel robot_model_;
  ik::CollisionChecker collision_checker_;
  IkTuning ik_tuning_;

  Eigen::VectorXd q_;
  // stepPickIk/stepPlaceIk가 실제로 쓰는 joint_weights - PICK/PLACE 진입(state_changed)
  // 시점에 ik_tuning_.joint_weights/place_joint_weights 중 해당하는 걸로 덮어씀.
  Eigen::VectorXd active_joint_weights_;
  // stepPickIk/stepPlaceIk가 실제로 쓰는 joint_weight_scale - PICK/VIRTUAL_PLACE 진입 시
  // ik_tuning_.joint_weight_scale로 세팅되지만, COMPLETE_PLACE 역재생(stepCompletePlaceHoming)
  // 진입 시엔 0.0으로 덮어씀(그냥 되돌아가는 구간이라 조인트 사용량을 아낄 필요가
  // 없어서 - active_joint_weights_와 동일 패턴).
  double active_joint_weight_scale_ = 0.0;
  double damping_ = ik::kDefaultDamping;
  Eigen::Vector3d target_pos_ = Eigen::Vector3d::Zero();
  bool has_captured_target_ = false;
  // VIRTUAL_PLACE 전용: isTiltSettled()가 처음 true가 되는 tick에 discardPendingVision()을
  // 한 번만 부르기 위한 플래그(solveTickImpl 참고) - tilt가 settle되는 동안 카메라가
  // 아직 이전 각도로 찍은 값이 vision_x_/y_/z_에 남아있다가 그대로 첫 표본으로 섞여
  // 들어가는 문제(실측 확인됨: 10개 중 1개만 나머지와 동떨어진 값)가 있었음.
  bool tilt_settle_vision_discarded_ = false;
  // tryCapturePendingVision이 논블로킹으로 모으는 비전 표본 개수(노이즈를 줄이려고
  // 평균냄).
  static constexpr int kVisionSampleTarget = 10;
  // tryCapturePendingVision이 tick 간에 이어서 누적하는 상태 - resetVisionSampleAccumulator()로
  // 새 캡처 시도 시작 전에 리셋해야 함(data_mutex_로 보호).
  Eigen::Vector3d vision_sample_sum_ = Eigen::Vector3d::Zero();
  int vision_sample_count_ = 0;
  // vision_sample_sum_과 동시에 채워지는 개별 표본(디버그 로그 전용) - kVisionSampleTarget개가
  // 다 모여 평균을 낼 때 전부 로그로 찍은 뒤 비움(resetVisionSampleAccumulator()에서도 비움).
  std::vector<Eigen::Vector3d> vision_samples_;
  // stepPickReady 전용: 팔이 home_q_full ready 자세에 settle되고 비전도 캡처된
  // 뒤에야 torso_yaw 편향(+-pick_ready_torso_bias_rad) 명령을 한 번 내보내고
  // true로 세팅 - 그 뒤로는 torso가 그 값에 실제로 도달했는지만 확인함.
  bool torso_bias_commanded_ = false;

  // stepPickReady 전용: PICK_READY에 최초로(SIT 직후) 진입한 그 순간의 실측
  // 자세를 딱 한 번 latch해두고, 그립 실패로 PICK_READY에 재진입할 때도 매번
  // 이 값을 그대로 재사용함 - 안 그러면(매번 그 순간 motor_position()을 다시
  // 읽으면) 그립 실패 직전 팔이 뻗어나가 있던 자세가 그대로 "홀드할 위치"가
  // 돼버려서 원래의 ready 자세로 다시는 안 돌아옴(onActivatedImpl에서 SIT 진입
  // 시에만 리셋 - 매 pick 사이클마다 새로 latch).
  bool pick_ready_q_latched_ = false;
  Eigen::Vector4d pick_ready_latched_q_ = Eigen::Vector4d::Zero();

  // PICK 전용: stepPickReady가 PICK_READY -> PICK 전이 직전(advance() 호출 바로
  // 앞)에 시작(EE)->target_pos_ 2점으로 한 번 만들어두고, PICK 상태(solveTickImpl)가
  // 매 틱 이 궤적의 result(pick_approach_traj_time_)를 target_pos_로 갱신해서
  // stepPickIk에 흘려보냄(place_trajectory_와 동일 패턴, IkTuning::pick_approach_duration_sec 참고).
  ik::TrajectoryGenerator pick_approach_trajectory_;
  double pick_approach_traj_time_ = 0.0;

  // VIRTUAL_PLACE 전용: target_pos_를 매 틱 이 궤적의 result(place_traj_time_)로 갱신해서
  // 기존 ikStep 루프에 그대로 흘려보냄(경유점은 stepPickIk/stepPlaceIk 진입 전에 구성).
  // COMPLETE_PLACE(stepCompletePlaceHoming)가 같은 place_trajectory_를 place_traj_time_을
  // duration에서 0으로 거꾸로 흘려보내며 재사용 - 놓으러 갈 때 지나간 장애물 회피
  // 아크를 그대로 되짚어 돌아오게 함(clear()는 다음 VIRTUAL_PLACE 재진입 때만 호출됨).
  ik::TrajectoryGenerator place_trajectory_;
  double place_traj_time_ = 0.0;
  bool ik_converged_ = false;
  int stuck_streak_ = 0;
  // VIRTUAL_PLACE 전용(stepPlaceIk 참고) - ik_tuning_.tol을 그대로 안 쓰고
  // 이 값을 params.tol로 씀(PICK은 ik_tuning_.tol 그대로). kPlaceTolInitialM(3mm)로
  // 시작해서, 연속 10번 수렴하면 kPlaceTolStepM(3mm)씩 조이되(하한 ik_tuning_.tol),
  // 연속 10번 미수렴하면 3mm씩 풀어줌(상한 없음).
  double place_tol_ = 0.0;  // 0.0이면 "아직 초기화 안 됨" - state_changed에서 세팅.
  int place_tol_success_streak_ = 0;
  int place_tol_fail_streak_ = 0;
  // VIRTUAL_PLACE 놓기 전용(stepPlaceIk 참고) - 그리퍼를 벌리기 시작하는 순간
  // 딱 한 번만 그 시점 실측 left_shoulder_pitch + offset을 q_[kLeftArmIndices[0]]에
  // 바로 대입해서 고정해둠(매 tick 다시 읽으면 목표 자체가 계속 움직여서 isSettled가
  // 영영 안 됨) - 목표값 자체는 q_ 안에 있으므로 별도 필드로 안 둠.
  bool place_release_shoulder_pitch_commanded_ = false;
  // 위와 동일 이유(elbow_pitch 버전) - place_release_elbow_offset_rad 필드 주석 참고.
  bool place_release_elbow_commanded_ = false;
  // 그리퍼를 열기 전 Cartesian 안전판(place_release_max_error_m 필드 주석,
  // stepPlaceIk 참고)이 "한 번이라도" real_error <= place_release_max_error_m를
  // 확인하면 true로 래치됨 - 그 뒤로는 매 tick 다시 검사 안 하고 그리퍼 오픈
  // 시퀀스(어깨/팔꿈치 오프셋 포함)를 계속 진행함. 매 tick 계속 재검사하면, 그
  // 오프셋 자체가 일부러 EE를 target_pos_에서 떨어뜨리는 동작이라 오프셋 적용 후
  // real_error가 다시 max를 넘어서 이 가드가 q_를 되돌려버리는 충돌이 있었음
  // (대화 참고 - 그리퍼 여는 동안 팔이 흔들리던 원인).
  bool place_release_error_confirmed_ok_ = false;
  bool kick_settling_ = false;
  double gripper_position_ = 0.0;

  Phase prev_solve_phase_;
  PickState prev_solve_pick_state_;
  PlaceState prev_solve_place_state_;

  // 그립 위글 시퀀스(COMPLETE_GRIP 모션 종료 후 kick->restore, 각 단계는
  // isSettled()로 확인).
  bool pending_grip_wiggle_ = false;
  bool grip_wiggle_kicked_ = false;
  bool grip_wiggle_restore_commanded_ = false;
  // grip_wiggle_restore_commanded_가 true가 된 뒤 isSettled 대기 중 흐른 틱 수 -
  // ik_tuning_.grip_wiggle_restore_timeout_ticks 타임아웃 판정용.
  int grip_wiggle_restore_wait_ticks_ = 0;
  double cached_grip_position_ = 0.0;

  // makeActionSnapshot()이 play_motion_number를 실어 보내면 true로 세팅되고,
  // onMotionEndImpl()이 motion_end를 처리하는 즉시 false로 풀림 - 모션이 재생되는
  // 동안엔(어느 step 함수가 재생했든) isSettled()가 무조건 false를 주게 해서, 외부
  // 모션 재생기가 모터를 구동 중인 시점에 우리 목표값 기준 settle 판정을 하지 않게 함.
  bool motion_in_flight_ = false;

  // makeActionSnapshot()/finalize()가 매번 새로 만들지 않고 계속 재사용하는 액션.
  FsmAction action_;
};
