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

TEST(ManipulationFsm, PickReadySendsArmGoalImmediatelyThenWaitsArmSettleAndVision) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);  // SIT -> PICK_READY

  // 1단계: state_changed 틱은 비전 여부와 무관하게 팔(shoulder_pitch/roll/elbow)
  // ready 자세 명령을 바로 내보냄 - torso_yaw는 아직 안 건드림.
  const FsmAction tick1 = fsm->solve_tick();
  ASSERT_TRUE(tick1.publish_motor_command);
  EXPECT_EQ(tick1.state_string, "PICK.PICK_READY");
  EXPECT_NEAR(tick1.motor_goal_positions.at("0"), -2.3, 1e-9);
  EXPECT_NEAR(tick1.motor_goal_positions.at("2"), 1.0, 1e-9);
  EXPECT_NEAR(tick1.motor_goal_positions.at("4"), 1.4, 1e-9);

  // 팔이 아직 ready 자세에 도달하지 않았으면(속도 남아있음) 계속 대기.
  fsm->set_motor_position("0", -1.8);
  fsm->set_motor_velocity("0", 50.0);
  const FsmAction still_moving = fsm->solve_tick();
  EXPECT_EQ(still_moving.state_string, "PICK.PICK_READY");

  // 팔(+ torso도 아직 안 건드렸으므로 home_q_full 프리셋과 동일한 위치)이 전부
  // settle됐다고 피드백해도, 비전이 아직 없으면 조용히 대기(publish 없음).
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);
  const FsmAction waiting_for_vision = fsm->solve_tick();
  EXPECT_FALSE(waiting_for_vision.publish_motor_command);
  EXPECT_EQ(waiting_for_vision.state_string, "PICK.PICK_READY");
}

TEST(ManipulationFsm, PickReadyBiasesTorsoTowardVisionYOnceArmSettledThenAdvancesOnceTorsoSettled) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);
  fsm->solve_tick();  // state_changed: 팔 ready 명령만 나감

  // 팔이 ready 자세에 도달했다고 피드백.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);

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
  fsm->on_motion_end(true);
  fsm->solve_tick();  // state_changed: 팔 ready 명령만 나감

  // 팔이 ready 자세에 도달했다고 피드백.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);

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

TEST(ManipulationFsm, CompleteGripReturnsLeftArmHomeThenPlaysMotionOnceSettled) {
  // motor id -> 기대값. torso("22")는 home_q_full의 값(0.0) 그대로 쓰고, 나머지
  // 팔 관절은 stepCompleteGripHoming 전용 IkTuning 기본값(complete_grip_*)을 씀 -
  // home_q_full(PICK_READY용)과는 별개(공을 든 채로 안전하게 들고 있을 자세).
  const std::map<std::string, double> home_by_id = {
      {"22", 0.0}, {"0", -2.1981}, {"2", 0.0}, {"4", -1.4}};

  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP

  // 첫 tick: 왼팔 홈 복귀 명령만 나가고, 아직 모터 피드백이 없어 settle 전이라
  // 모션은 아직 재생 안 함.
  const FsmAction first = fsm->solve_tick();
  ASSERT_TRUE(first.publish_motor_command);
  EXPECT_FALSE(first.play_motion_number.has_value());
  for (const auto& [id, home_value] : home_by_id) {
    EXPECT_NEAR(first.motor_goal_positions.at(id), home_value, 1e-9);
  }

  // 아직 모터가 안 도달했다고 보고하면(속도 남아있음) 계속 대기.
  for (const auto& [id, home_value] : home_by_id) {
    fsm->set_motor_position(id, home_value + 0.5);
    fsm->set_motor_velocity(id, 50.0);
  }
  const FsmAction still_moving = fsm->solve_tick();
  EXPECT_FALSE(still_moving.play_motion_number.has_value());

  // 전부 도달(home 값 근처 + 속도 0)했다고 보고하면 그제서야 모션(74) 재생.
  for (const auto& [id, home_value] : home_by_id) {
    fsm->set_motor_position(id, home_value);
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
  // place_traj_time_이 거꾸로 0까지 흘러가야 함 - 그동안은 모터 피드백을 하나도
  // 안 줬으니 isSettled가 전부 false라 모션(76)이 재생될 수 없음. 마지막 tick의
  // motor_goal_positions가 역재생 IK의 최종 수렴 목표(torso+왼팔)임.
  FsmAction last_action;
  for (int i = 0; i < 200; ++i) {
    last_action = fsm->solve_tick();
    EXPECT_FALSE(last_action.play_motion_number.has_value());
    EXPECT_FALSE(last_action.deactivate_and_notify);
  }
  ASSERT_TRUE(last_action.publish_motor_command);

  // 역재생 IK가 수렴한 목표(torso+왼팔) 그대로 피드백을 주면 그제서야 settle로
  // 보고 모션(76)을 재생함.
  const std::vector<std::string> kIkMotorIds = {"0", "2", "4", "22"};
  for (const std::string& id : kIkMotorIds) {
    fsm->set_motor_position(id, last_action.motor_goal_positions.at(id));
    fsm->set_motor_velocity(id, 0.0);
  }
  const FsmAction settled = fsm->solve_tick();
  ASSERT_TRUE(settled.play_motion_number.has_value());
  EXPECT_EQ(*settled.play_motion_number, 76);
  EXPECT_FALSE(settled.deactivate_and_notify);

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
