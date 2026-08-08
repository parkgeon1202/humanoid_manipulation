#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>

#include "fsm.hpp"

namespace {

IkTuning makeTestTuning() {
  IkTuning tuning;
  tuning.tol = 1e-4;
  tuning.alpha = 0.5;
  tuning.joint_weight_scale = 0.005;
  tuning.secondary_gain = 0.01;
  tuning.collision_avoidance_threshold_m = 0.001;
  tuning.collision_avoidance_k_pull = 1.0;
  tuning.stuck_streak_ticks = 30;
  tuning.torso_kick_step_rad = 0.5;
  tuning.gripper_open_rad = 0.5;
  tuning.gripper_closed_rad = 0.0;
  tuning.left_shoulder_roll_target_rad = 1.0;
  tuning.settle_velocity_threshold = 0.3;
  tuning.settle_position_tolerance_rad = 0.05;
  tuning.grip_moving_velocity_threshold = 0.5;
  tuning.grip_near_zero_position_threshold = 0.1;
  tuning.grip_wiggle_kick_position_rad = 1.0;

  // 실제 yaml 기본값(pick_ready_home_q: [0, -2.3, 1, 1.4])과 동일하게 맞춤 - PICK_READY가
  // 더 이상 torso_yaw를 별도로 회전시키지 않고 이 자세를 그대로 IK 시작점으로 씀.
  tuning.home_q_full = Eigen::VectorXd::Zero(8);
  tuning.home_q_full[ik::kTorsoJointIndex] = 0.0;
  tuning.home_q_full[ik::kLeftArmIndices[0]] = -2.3;
  tuning.home_q_full[ik::kLeftArmIndices[1]] = 1.0;
  tuning.home_q_full[ik::kLeftArmIndices[2]] = 1.4;
  tuning.joint_weights = Eigen::VectorXd::Ones(8);
  tuning.joint_weights[1] = 10.0;
  tuning.joint_weights[2] = 10.0;
  tuning.joint_weights[3] = 10.0;
  // PLACE 전용 joint_weights도 채워둠(안 채우면 크기 0이라 VIRTUAL_PLACE에서
  // stepReachableIk가 active_joint_weights_를 쓸 때 크기가 안 맞아 터짐).
  tuning.place_joint_weights = Eigen::VectorXd::Ones(8);
  tuning.place_joint_weights[0] = 10.0;
  return tuning;
}

// ManipulationFSM은 복사/이동이 막혀 있음(collision_checker_가 robot_model_을
// 참조로 들고 있어서) - factory가 값으로 반환할 수 없으므로 unique_ptr로 감쌈.
std::unique_ptr<ManipulationFSM> makeFsm() {
  auto fsm = std::make_unique<ManipulationFSM>(TEST_URDF_PATH);
  fsm->configure_ik(makeTestTuning());
  return fsm;
}

TEST(ManipulationFsm, OnActivatedRequestsSitMotionOnlyAtSit) {
  auto fsm = makeFsm();
  const FsmAction action = fsm->on_activated();
  ASSERT_TRUE(action.play_motion_number.has_value());
  EXPECT_EQ(*action.play_motion_number, 73);
  EXPECT_EQ(action.state_string, "PICK.SIT");
}

TEST(ManipulationFsm, SolveTickIsNoOpDuringSit) {
  auto fsm = makeFsm();
  const FsmAction action = fsm->solve_tick();
  EXPECT_FALSE(action.publish_motor_command);
  EXPECT_FALSE(action.deactivate_and_notify);
}

TEST(ManipulationFsm, MotionEndAdvancesSitToPickReady) {
  auto fsm = makeFsm();
  const FsmAction action = fsm->on_motion_end(true);
  EXPECT_EQ(action.state_string, "PICK.PICK_READY");
  EXPECT_TRUE(action.publish_motor_command);
}

TEST(ManipulationFsm, OnActivatedRequestsPlaceMotionAtDone) {
  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE

  const FsmAction action = fsm->on_activated();
  ASSERT_TRUE(action.play_motion_number.has_value());
  EXPECT_EQ(*action.play_motion_number, 75);
  EXPECT_EQ(action.state_string, "PICK.DONE");
}

TEST(ManipulationFsm, MotionEndAdvancesDoneToVirtualPlaceAndLatchesLegsAndOtherArm) {
  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE

  // 다리 모터 하나("10")와 반대팔 모터 하나("1")가 지금 이 위치에 있다고 가정 -
  // place 진입 모션이 끝나면 이 값 그대로 latch돼야 함.
  fsm->set_motor_position("10", 0.42);
  fsm->set_motor_position("1", -0.17);

  const FsmAction action = fsm->on_motion_end(true);
  EXPECT_EQ(action.state_string, "PLACE.VIRTUAL_PLACE");
  ASSERT_TRUE(action.publish_motor_command);
  EXPECT_NEAR(action.motor_goal_positions.at("10"), 0.42, 1e-9);
  EXPECT_NEAR(action.motor_goal_positions.at("1"), -0.17, 1e-9);
}

TEST(ManipulationFsm, PickReadyHoldsLatchedPositionAndCapturesVisionOnceSettled) {
  auto fsm = makeFsm();
  // state_changed 이전에 모터가(SIT 모션이 끝난 실제 자세로) 임의 위치에 있고,
  // 이미 그 자리에 정지해 있다고(속도 0) 가정 - 더 이상 home_q_full로 옮기지
  // 않고 이 값을 최초 latch로 그대로 q_에 읽어와야 함.
  fsm->set_motor_position("22", 0.2);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -1.0);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 0.4);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 0.9);
  fsm->set_motor_velocity("4", 0.0);
  fsm->on_motion_end(true);  // SIT -> PICK_READY

  // 1단계: state_changed 틱은 home_q_full 프리셋이 아니라 방금 읽어 latch해둔
  // 실측 위치를 그대로 홀드 명령으로 내보냄.
  const FsmAction tick1 = fsm->solve_tick();
  ASSERT_TRUE(tick1.publish_motor_command);
  EXPECT_EQ(tick1.state_string, "PICK.PICK_READY");
  EXPECT_NEAR(tick1.motor_goal_positions.at("22"), 0.2, 1e-9);
  EXPECT_NEAR(tick1.motor_goal_positions.at("0"), -1.0, 1e-9);
  EXPECT_NEAR(tick1.motor_goal_positions.at("2"), 0.4, 1e-9);
  EXPECT_NEAR(tick1.motor_goal_positions.at("4"), 0.9, 1e-9);

  // 이미 그 자리에 정지해 있으므로 settle 체크가 바로 통과해서, 비전이 아직
  // 없으면(publish 없이) 곧장 비전 대기로 넘어감 - 추가 이동 없이 즉시 통과되는지
  // 확인.
  const FsmAction waiting_for_vision = fsm->solve_tick();
  EXPECT_FALSE(waiting_for_vision.publish_motor_command);
  EXPECT_EQ(waiting_for_vision.state_string, "PICK.PICK_READY");
}

TEST(ManipulationFsm, PickReadyRetryReusesFirstLatchedPositionAndWaitsForRealSettle) {
  // 그립 실패로 PICK_READY에 재진입할 때, 그 순간의 실측값(뻗어나가 있던 실패
  // 자세)을 다시 latch하는 게 아니라 최초 진입 때의 latch를 그대로 재사용해야
  // 하고, 실제로 그 자리에 도달할 때까지는 비전 캡처를 시작하면 안 됨
  // (pick_ready_latched_q_ 필드 주석 참고).
  auto fsm = makeFsm();
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);
  fsm->on_motion_end(true);  // SIT -> PICK_READY (여기서 위 값이 최초 latch)
  fsm->solve_tick();  // state_changed

  // 비전 target을 일부러 latch된 자세의 FK 지점과 정확히 같게 잡음(x=0.1765,
  // z=0.0635, y는 vision_y_offset(0.0328)을 상쇄하도록 0.2323 -> 0.1995) - PICK
  // approach가 사실상 제자리 궤적이 되므로 큰 변위를 실제로 추적시킬 필요 없이
  // pick_approach_duration_sec(1.0s=50tick)가 끝나자마자 바로 수렴함(이 테스트의
  // 목적은 PICK IK 자체의 수렴성이 아니라 아래의 latch/retry 로직 검증임). y가
  // 0.02 이상이라 torso 재명령 분기를 타긴 하지만 양수 쪽은 *0.0이라(코드 주석
  // 참고) 실질적으로 값이 안 바뀜.
  for (int i = 0; i < 10; ++i) {
    fsm->update_vision(0.1765, 0.2323, 0.0635);
    fsm->solve_tick();
  }
  // 10번째 tick에서 비전 캡처+torso_bias_commanded_가 끝나고, 다음 tick에서
  // stage-2 settle 확인(이미 그 자리라 즉시 통과)까지 거쳐 PICK으로 전이함.
  ASSERT_EQ(fsm->solve_tick().state_string, "PICK.PICK");

  // pick_approach_trajectory_(duration 1.0 / dt 0.02 = 50tick)가 다 끝나고 IK가
  // 실제로 tol 안으로 수렴할 때까지 tick을 충분히 돌림 - ik_converged_가 true여야만
  // checkGripMotorStopped가 그립 정지 판정을 시작함.
  for (int i = 0; i < 200; ++i) {
    fsm->solve_tick();
  }

  // PICK 도중 팔이 최초 latch와는 다른 자세(뻗어나간 자세)로 옮겨갔다가, 그립
  // 실패로 PICK_READY로 되돌아왔다고 가정(checkGripMotorStopped와 동일한 방식 -
  // 그립 모터가 한 번 빠르게 움직인 뒤 gripper_closed_rad 근처에서 멈추면 재시도).
  fsm->set_motor_position("22", 0.9);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -1.5);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 0.2);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 0.5);
  fsm->set_motor_velocity("4", 0.0);
  fsm->set_motor_velocity("6", 5.0);  // 그립 모터가 닫히는 중(빠르게 움직임)
  fsm->on_motor_feedback();
  fsm->set_motor_position("6", 0.0);  // gripper_closed_rad(0.0) 그대로 - 아무것도 못 집음
  fsm->set_motor_velocity("6", 0.0);
  FsmAction retry_action;
  for (int i = 0; i < 10; ++i) {  // grip_stopped_streak_ticks 기본값(10)
    retry_action = fsm->on_motor_feedback();
  }
  ASSERT_EQ(retry_action.state_string, "PICK.PICK_READY");

  // 재진입 직후(state_changed) 틱: 방금 실측한 뻗은 자세(0.9/-1.5/0.2/0.5)가
  // 아니라 최초 latch값(0.0/-2.3/1.0/1.4) 그대로 홀드 명령이 나가야 함.
  const FsmAction retry_hold = fsm->solve_tick();
  ASSERT_TRUE(retry_hold.publish_motor_command);
  EXPECT_NEAR(retry_hold.motor_goal_positions.at("22"), 0.0, 1e-9);
  EXPECT_NEAR(retry_hold.motor_goal_positions.at("0"), -2.3, 1e-9);
  EXPECT_NEAR(retry_hold.motor_goal_positions.at("2"), 1.0, 1e-9);
  EXPECT_NEAR(retry_hold.motor_goal_positions.at("4"), 1.4, 1e-9);

  // 아직 실측이 latch값에 도달 전(방금 세팅한 뻗은 자세 그대로)이므로, 비전을
  // 줘도 settle 게이트에 막혀 캡처가 시작되면 안 됨(publish_motor_command은
  // 계속 true - 홀드 명령을 계속 재전송하며 대기).
  fsm->update_vision(0.1765, 0.2323, 0.0635);
  const FsmAction still_settling = fsm->solve_tick();
  EXPECT_TRUE(still_settling.publish_motor_command);
  EXPECT_EQ(still_settling.state_string, "PICK.PICK_READY");

  // 실제로 latch값에 도달했다고 피드백하면 그제서야 비전 캡처가 시작되고,
  // 10개 표본을 다 채우면 다시 PICK으로 전이함.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_position("4", 1.4);
  for (int i = 0; i < 10; ++i) {
    fsm->update_vision(0.1765, 0.2323, 0.0635);
    fsm->solve_tick();
  }
  const FsmAction entered_pick_again = fsm->solve_tick();
  EXPECT_EQ(entered_pick_again.state_string, "PICK.PICK");
}

TEST(ManipulationFsm, PickReadyBiasesTorsoTowardVisionYOnceArmSettledThenAdvancesOnceTorsoSettled) {
  auto fsm = makeFsm();
  // SIT 끝난 시점에 팔이 이미 이 자세(정지)에 있다고 가정 - state_changed가 이
  // 값을 최초 latch로 읽어들이므로, settle 게이트가 바로 통과함.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);
  fsm->on_motion_end(true);
  fsm->solve_tick();  // state_changed: 팔 ready 명령만 나감(latch=현재 실측값)

  // 팔 settle이 확인된 뒤에야 비전 캡처가 실제로 쓰임 - tryCapturePendingVision이
  // kVisionSampleTarget(10)개를 모아 평균낼 때까지 캡처를 안 끝내므로, 9번은
  // 아직 대기 상태여야 함.
  for (int i = 0; i < 9; ++i) {
    fsm->update_vision(0.05, -0.15, -0.15);
    const FsmAction still_waiting = fsm->solve_tick();
    EXPECT_FALSE(still_waiting.publish_motor_command);
  }
  // 10번째 표본으로 평균 캡처 완료 - y가 음수(오른쪽)이므로 torso_yaw를
  // -pick_ready_torso_bias_rad(기본 1.0, 오른쪽은 그대로 -1.0)로 명령해야 함
  // (왼쪽/양수 쪽은 지금 *0.0이라 이 분기로는 의미 있는 검증이 안 됨).
  fsm->update_vision(0.05, -0.15, -0.15);
  const FsmAction torso_commanded = fsm->solve_tick();
  ASSERT_TRUE(torso_commanded.publish_motor_command);
  EXPECT_EQ(torso_commanded.state_string, "PICK.PICK_READY");  // 아직 PICK 아님
  EXPECT_NEAR(torso_commanded.motor_goal_positions.at("22"), -1.0, 1e-9);

  // torso가 아직 그 값에 도달하지 않았으면 PICK_READY에 계속 머무름.
  const FsmAction torso_still_moving = fsm->solve_tick();
  EXPECT_EQ(torso_still_moving.state_string, "PICK.PICK_READY");

  // torso가 실제로 편향값(-1.0)에 도달했다고 피드백하면 그제서야 PICK으로 전이.
  fsm->set_motor_position("22", -1.0);
  fsm->set_motor_velocity("22", 0.0);
  const FsmAction entered_pick = fsm->solve_tick();
  EXPECT_EQ(entered_pick.state_string, "PICK.PICK");
}

TEST(ManipulationFsm, PickReadyDiscardsAllZeroAndSentinelVisionSamplesFromAverage) {
  auto fsm = makeFsm();
  // SIT 끝난 시점에 팔이 이미 이 자세(정지)에 있다고 가정 - state_changed가 이
  // 값을 최초 latch로 읽어들이므로, settle 게이트가 바로 통과함.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);
  fsm->on_motion_end(true);
  fsm->solve_tick();  // state_changed: 팔 ready 명령만 나감(latch=현재 실측값)

  // x/y/z 셋 다 0.0이거나 셋 다 -0.999(센티널)인 표본, 축 하나만 -0.999인 부분
  // 센티널 표본(z만 깊이 실패 등 - 실측에서 확인된 케이스), 그리고 셋 다 magnitude
  // bound(0.33)를 넘는 표본(공/PICK_READY 캡처 전용)은 tryCapturePendingVision이
  // 평균에서 버려야 함 - 아무리 많이 보내도 카운트가 안 늘어서 계속 대기 상태여야 함.
  for (int i = 0; i < 30; ++i) {
    double x = 0.0, y = 0.0, z = 0.0;
    if (i % 5 == 1) {
      x = y = z = -0.999;  // 셋 다 센티널
    } else if (i % 5 == 2) {
      x = y = z = 0.5;  // magnitude bound(0.33) 초과
    } else if (i % 5 == 3) {
      x = -0.999; y = 0.05; z = -0.15;  // x만 부분 센티널(y/z는 정상값)
    } else if (i % 5 == 4) {
      x = 0.05; y = -0.15; z = -0.999;  // z만 부분 센티널(x/y는 정상값)
    }
    fsm->update_vision(x, y, z);
    const FsmAction still_waiting = fsm->solve_tick();
    EXPECT_FALSE(still_waiting.publish_motor_command);
  }

  // 버려진 표본은 카운트에 안 들어갔어야 하므로, 이제부터 유효한 표본 10개를
  // 채워야 캡처가 끝남(9개까진 계속 대기).
  for (int i = 0; i < 9; ++i) {
    fsm->update_vision(0.05, -0.15, -0.15);
    const FsmAction still_waiting = fsm->solve_tick();
    EXPECT_FALSE(still_waiting.publish_motor_command);
  }
  fsm->update_vision(0.05, -0.15, -0.15);
  const FsmAction torso_commanded = fsm->solve_tick();
  ASSERT_TRUE(torso_commanded.publish_motor_command);
  EXPECT_NEAR(torso_commanded.motor_goal_positions.at("22"), -1.0, 1e-9);
}

TEST(ManipulationFsm, CompleteGripPullsBackXThenPlaysMotionOnceSettled) {
  // stepCompleteGripHoming은 더 이상 고정 프리셋으로 스냅하지 않고, 집은 자세(q_)의
  // FK EE 위치에서 x를 complete_grip_pull_back_x_m(0.1)만큼 뺀 지점으로 IK를 품.
  // advance()만으로 건너뛰면 q_가 생성자 기본값(Zero(nq) - 팔이 쭉 펴진 채 특이점에
  // 가까운 비현실적 자세)에 머물러서, active_joint_weight_scale_=0.0(정규화 없음)과
  // 겹쳐 damping이 상한까지 튀어 다시는 못 내려오는 채로 영영 안 수렴하는 인공적인
  // 교착이 생김 - 실제로는 이 상태에 PICK이 IK로 수렴해둔(즉 특이점과 거리가 있는)
  // 자세로 진입하므로, 테스트도 PICK_READY의 모터 피드백을 home_q_full과 동일한
  // 값으로 줘서 q_를 그 현실적인 자세로 채워둔 뒤 진입시킴.
  const std::vector<std::string> kIkMotorIds = {"0", "2", "4", "22"};

  auto fsm = makeFsm();
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_position("4", 1.4);
  fsm->on_motion_end(true);  // SIT -> PICK_READY
  fsm->solve_tick();  // state_changed: 위 실측값을 그대로 q_로 읽어들임
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP

  // 모터 피드백을 전혀 안 주는 동안은 isSettled가 항상 false라 모션이 재생될 수
  // 없음 - 그동안 IK가 매 tick 계속 수렴을 시도함.
  FsmAction last_action;
  for (int i = 0; i < 200; ++i) {
    last_action = fsm->solve_tick();
    EXPECT_FALSE(last_action.play_motion_number.has_value());
  }
  ASSERT_TRUE(last_action.publish_motor_command);

  // IK가 수렴한 목표(torso+왼팔) 그대로 피드백을 주면 그제서야 settle로 보고
  // 모션(74)을 재생함.
  for (const std::string& id : kIkMotorIds) {
    fsm->set_motor_position(id, last_action.motor_goal_positions.at(id));
    fsm->set_motor_velocity(id, 0.0);
  }
  const FsmAction settled = fsm->solve_tick();
  ASSERT_TRUE(settled.play_motion_number.has_value());
  EXPECT_EQ(*settled.play_motion_number, 74);

  // settled인 채로 tick이 계속 돌아도(모션이 아직 안 끝났으므로) 모션을 또
  // 재생 요청하면 안 됨.
  const FsmAction still_settled = fsm->solve_tick();
  EXPECT_FALSE(still_settled.play_motion_number.has_value());
}

TEST(ManipulationFsm, DoneHomesLeftArmOnceThenWaitsForSettleBeforeDeactivating) {
  // stepDoneHoming()은 home_q_full이 아니라 항상 0.0으로 복귀시킴(motor id -> 0.0).
  const std::map<std::string, double> home_by_id = {
      {"22", 0.0}, {"0", 0.0}, {"2", 0.0}, {"4", 0.0}};

  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE

  // 첫 tick: 왼팔 홈 복귀 명령만 나가고, 아직 settle 전이라 deactivate 안 함.
  const FsmAction first = fsm->solve_tick();
  ASSERT_TRUE(first.publish_motor_command);
  EXPECT_FALSE(first.deactivate_and_notify);
  for (const auto& [id, home_value] : home_by_id) {
    EXPECT_NEAR(first.motor_goal_positions.at(id), home_value, 1e-9);
  }

  // 아직 모터가 안 도달했다고 보고하면(속도 남아있음) 계속 대기.
  for (const auto& [id, home_value] : home_by_id) {
    fsm->set_motor_position(id, home_value + 0.5);
    fsm->set_motor_velocity(id, 50.0);
  }
  const FsmAction still_moving = fsm->solve_tick();
  EXPECT_FALSE(still_moving.deactivate_and_notify);

  // 전부 도달(home 값 근처 + 속도 0)했다고 보고하면 그제서야 deactivate.
  for (const auto& [id, home_value] : home_by_id) {
    fsm->set_motor_position(id, home_value);
    fsm->set_motor_velocity(id, 0.0);
  }
  const FsmAction settled = fsm->solve_tick();
  EXPECT_TRUE(settled.deactivate_and_notify);
}

TEST(ManipulationFsm, VirtualPlaceEntersReachableIkImmediatelyWithoutRotationPhase) {
  auto fsm = makeFsm();
  // advance()로 PICK_READY/PICK/COMPLETE_GRIP/DONE을 건너뛰고 곧장 PLACE.VIRTUAL_PLACE로.
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE
  ASSERT_TRUE(fsm->advance());  // DONE -> PLACE.VIRTUAL_PLACE
  ASSERT_EQ(fsm->phase(), Phase::PLACE);
  ASSERT_EQ(fsm->place_state(), PlaceState::VIRTUAL_PLACE);

  // advance()로 건너뛰어서 q_는 생성자 기본값(Zero(nq))에 머물러 있음 - place_trajectory_
  // 추적을 시작하기 전에 그 자리에 실제로 도달했는지 확인하는 게이트가 있으므로,
  // 이미 그 값(0.0)에 정지해 있다고 실측 피드백을 미리 줘서 게이트가 바로 통과되게 함.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", 0.0);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 0.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 0.0);
  fsm->set_motor_velocity("4", 0.0);

  // 회전 준비 단계 없이, 비전을 주면 바로 도달 가능 IK(hard/soft 충돌 회피 포함)가
  // 도는 solve_tick 결과가 나와야 함 - 허리를 별도로 1.5로 고정하는 로직이 없으므로
  // torso 목표가 1.5로 강제되지 않음(PICK_READY와의 핵심 차이). tryCapturePendingVision이
  // kVisionSampleTarget(10)개를 모아 평균낼 때까지 캡처를 안 끝내므로 10번 반복.
  for (int i = 0; i < 9; ++i) {
    fsm->update_vision(0.02, -0.05, -0.25);
    fsm->solve_tick();
  }
  fsm->update_vision(0.02, -0.05, -0.25);
  const FsmAction action = fsm->solve_tick();
  ASSERT_TRUE(action.publish_motor_command);
  EXPECT_NE(action.motor_goal_positions.at("22"), -1.5);
}

TEST(ManipulationFsm, VirtualPlaceVisionCaptureIgnoresMagnitudeBound) {
  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE
  ASSERT_TRUE(fsm->advance());  // DONE -> PLACE.VIRTUAL_PLACE

  // x/y/z 절댓값이 전부 kVisionMagnitudeBound(0.33)를 넘는 값 - 공(PICK_READY)
  // 캡처였다면 버려졌겠지만, 골대(VIRTUAL_PLACE) 캡처는 check_magnitude_bound=false로
  // 호출하므로 이 범위 제한 없이 그대로 캡처돼야 함.
  for (int i = 0; i < 9; ++i) {
    fsm->update_vision(0.5, -0.5, -0.5);
    fsm->solve_tick();
  }
  fsm->update_vision(0.5, -0.5, -0.5);
  const FsmAction action = fsm->solve_tick();
  ASSERT_TRUE(action.publish_motor_command);
}

TEST(ManipulationFsm, CompletePlaceHomingReversesTrajectoryThenPlaysMotion76BeforeDeactivating) {
  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE
  fsm->on_activated();

  // advance()로 건너뛰지 않고 실제로 on_motion_end(true)를 불러야 함 - on_activated()가
  // 방금 재생한 place 진입 모션(75) 때문에 motion_in_flight_가 true로 세팅됐는데,
  // 이걸 풀어주는 게 on_motion_end()뿐이라(advance()는 이 플래그를 안 건드림) 안
  // 부르면 이후 isSettled()가 계속 무조건 false로 나와서 settle 판정이 영영 안 됨.
  const FsmAction after_place_entry_motion = fsm->on_motion_end(true);  // DONE -> PLACE.VIRTUAL_PLACE
  ASSERT_EQ(after_place_entry_motion.state_string, "PLACE.VIRTUAL_PLACE");

  // advance()로 건너뛰어서 q_는 생성자 기본값(Zero(nq))에 머물러 있음 - VIRTUAL_PLACE/
  // COMPLETE_PLACE 둘 다 궤적 추적을 시작하기 전에 그 자리에 실제로 도달했는지
  // 확인하는 게이트가 있으므로, 이미 그 값(0.0)에 정지해 있다고 실측 피드백을
  // 미리 줘서 게이트가 바로 통과되게 함.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", 0.0);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 0.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 0.0);
  fsm->set_motor_velocity("4", 0.0);

  // COMPLETE_PLACE가 거꾸로 재생할 place_trajectory_가 실제로 있어야 하므로, 먼저
  // 비전을 캡처시켜 VIRTUAL_PLACE가 진짜 궤적을 만들게 함
  // (VirtualPlaceEntersReachableIkImmediatelyWithoutRotationPhase와 동일 패턴).
  for (int i = 0; i < 10; ++i) {
    fsm->update_vision(0.02, -0.05, -0.25);
    fsm->solve_tick();
  }

  ASSERT_TRUE(fsm->advance());  // VIRTUAL_PLACE -> PLACE.COMPLETE_PLACE
  ASSERT_EQ(fsm->place_state(), PlaceState::COMPLETE_PLACE);

  // place_traj_duration_sec(3.0)/place_traj_dt_sec(0.02) ~= 150tick에 걸쳐
  // place_traj_time_이 거꾸로 0까지 흘러가야 함. COMPLETE_PLACE도 VIRTUAL_PLACE와
  // 동일하게 매 스텝 실제 도달을 확인한 뒤에야 다음 t로 넘어가는 게이트가 있으므로,
  // 매 tick 방금 명령한 목표를 그대로(즉시 도달했다고) 피드백해줘서 게이트가 매번
  // 통과되게 함(실제 하드웨어 동역학은 이 테스트의 관심사가 아님) - 이 루프
  // 자체가 매 tick 실제로 "도달"까지 시뮬레이션하므로, 역재생이 끝나 모션(76)
  // 재생이 요청되는 순간까지 자연스럽게 도달함.
  const std::vector<std::string> kIkMotorIds = {"0", "2", "4", "22"};
  FsmAction action;
  bool motion_requested = false;
  for (int i = 0; i < 300 && !motion_requested; ++i) {
    action = fsm->solve_tick();
    motion_requested = action.play_motion_number.has_value();
    EXPECT_FALSE(action.deactivate_and_notify);
    if (action.publish_motor_command) {
      for (const std::string& id : kIkMotorIds) {
        fsm->set_motor_position(id, action.motor_goal_positions.at(id));
        fsm->set_motor_velocity(id, 0.0);
      }
    }
  }
  ASSERT_TRUE(motion_requested);
  EXPECT_EQ(*action.play_motion_number, 76);
  EXPECT_FALSE(action.deactivate_and_notify);

  // settled인 채로 tick이 계속 돌아도(모션이 아직 안 끝났으므로) 모션을 또
  // 재생 요청하면 안 됨(CompleteGripReturnsLeftArmHomeThenPlaysMotionOnceSettled와 동일 패턴).
  const FsmAction still_settled = fsm->solve_tick();
  EXPECT_FALSE(still_settled.play_motion_number.has_value());
  EXPECT_FALSE(still_settled.publish_motor_command);

  // 모션(76)이 실제로 끝나면 시퀀스 전체의 마지막이므로 곧바로 deactivate.
  const FsmAction motion_done = fsm->on_motion_end(true);
  EXPECT_TRUE(motion_done.deactivate_and_notify);
}

}  // namespace
