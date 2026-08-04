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
  tuning.gripper_step_rad = 0.02;
  tuning.left_shoulder_roll_target_rad = 1.0;
  tuning.settle_velocity_threshold = 0.3;
  tuning.settle_position_tolerance_rad = 0.05;
  tuning.grip_moving_velocity_threshold = 0.3;
  tuning.grip_near_zero_position_threshold = 0.2;
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

TEST(ManipulationFsm, PickReadyHomeResetsThenCapturesVisionOnceThenAdvancesImmediately) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);  // SIT -> PICK_READY

  // 홈(torso=0, 왼팔은 이미 ready 자세: -2.3,1,1.4) 자세에 이미 도달해 있다고 피드백
  // - 비전 캡처는 이 4개 관절이 전부 settle된 뒤에만 시작되므로(그립 실패 재시도
  // 시 이전 자세가 남아있지 않게), 이 피드백 없이는 tick3에서 캡처가 안 됨.
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);

  // 첫 tick: 홈 복귀 명령만 나가고 아직 비전 캡처 전.
  const FsmAction tick1 = fsm->solve_tick();
  ASSERT_TRUE(tick1.publish_motor_command);
  EXPECT_NEAR(tick1.motor_goal_positions.at("22"), 0.0, 1e-9);
  EXPECT_NEAR(tick1.motor_goal_positions.at("0"), -2.3, 1e-9);
  EXPECT_EQ(tick1.state_string, "PICK.PICK_READY");

  // 비전이 아직 없으면 캡처도 전이도 안 함(publish 여부만 확인).
  const FsmAction tick2 = fsm->solve_tick();
  EXPECT_FALSE(tick2.publish_motor_command);

  // 비전이 들어오면 그 순간 값을 캡처하고, torso_yaw를 미리 고정 회전시키는 단계
  // 없이(예전엔 -1.5rad로 돌리고 settle을 기다렸음) 바로 PICK(도달 가능 IK)로
  // 전이함 - 팔은 이미 home_q_full로 ready 자세라 더 돌 필요가 없음.
  fsm->update_vision(0.05, 0.15, -0.15);
  const FsmAction tick3 = fsm->solve_tick();
  EXPECT_EQ(tick3.state_string, "PICK.PICK");
}

TEST(ManipulationFsm, PickPreservesHomeArmPoseWithUnrotatedTorsoAsIkStartingPoint) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);
  fsm->set_motor_position("22", 0.0);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("0", -2.3);
  fsm->set_motor_velocity("0", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  fsm->set_motor_position("4", 1.4);
  fsm->set_motor_velocity("4", 0.0);
  fsm->solve_tick();  // 홈 복귀

  fsm->update_vision(0.05, 0.15, -0.15);
  const FsmAction entered_pick = fsm->solve_tick();  // 캡처 + 바로 PICK 전이
  ASSERT_EQ(entered_pick.state_string, "PICK.PICK");

  // PICK 진입 직후 첫 IK tick: q_가 0으로 리셋되지 않고 home_q_full(팔은 ready
  // 값, torso_yaw=0)에서 그대로 시작함 - torso_yaw를 미리 돌려두지 않았으므로
  // 한 스텝만으로 크게 안 움직여야 함(예전엔 -1.5rad 근처에서 시작해서 -1.0
  // 이하였음 - 이제는 0 근처에서 시작).
  ASSERT_TRUE(entered_pick.publish_motor_command);
  EXPECT_NEAR(entered_pick.motor_goal_positions.at("22"), 0.0, 0.5);
}

TEST(ManipulationFsm, CompleteGripReturnsLeftArmHomeThenPlaysMotionOnceSettled) {
  // motor id -> home_q_full 기대값(makeTestTuning의 [torso, l_shoulder_pitch,
  // l_shoulder_roll, l_elbow_pitch] = [0,-2.3,1,1.4]과 동일한 매핑).
  const std::map<std::string, double> home_by_id = {
      {"22", 0.0}, {"0", -2.3}, {"2", 1.0}, {"4", 1.4}};

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
  // motor id -> home_q_full 기대값(makeTestTuning의 [torso, l_shoulder_pitch,
  // l_shoulder_roll, l_elbow_pitch] = [0,-2.3,1,1.4]과 동일한 매핑).
  const std::map<std::string, double> home_by_id = {
      {"22", 0.0}, {"0", -2.3}, {"2", 1.0}, {"4", 1.4}};

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
  // torso 목표가 1.5로 강제되지 않음(PICK_READY와의 핵심 차이).
  fsm->update_vision(0.02, -0.05, -0.25);
  const FsmAction action = fsm->solve_tick();
  ASSERT_TRUE(action.publish_motor_command);
  EXPECT_NE(action.motor_goal_positions.at("22"), -1.5);
}

TEST(ManipulationFsm, CompletePlaceHomingCommandsOnceThenWaitsForSettleBeforeDeactivating) {
  auto fsm = makeFsm();
  ASSERT_TRUE(fsm->advance());  // SIT -> PICK_READY
  ASSERT_TRUE(fsm->advance());  // PICK_READY -> PICK
  ASSERT_TRUE(fsm->advance());  // PICK -> COMPLETE_GRIP
  ASSERT_TRUE(fsm->advance());  // COMPLETE_GRIP -> DONE
  ASSERT_TRUE(fsm->advance());  // DONE -> PLACE.VIRTUAL_PLACE
  ASSERT_TRUE(fsm->advance());  // VIRTUAL_PLACE -> PLACE.COMPLETE_PLACE
  ASSERT_EQ(fsm->place_state(), PlaceState::COMPLETE_PLACE);

  // 첫 tick: 0.0 홈 명령이 나가지만, 아직 motor_status 피드백이 하나도 없어서
  // isSettled()가 전부 false(찾을 수 없음)로 처리 -> deactivate 아직 안 함.
  const FsmAction first = fsm->solve_tick();
  ASSERT_TRUE(first.publish_motor_command);
  for (const std::string& id : {std::string("0"), std::string("2"), std::string("4"),
                                 std::string("22")}) {
    EXPECT_NEAR(first.motor_goal_positions.at(id), 0.0, 1e-9);
  }
  EXPECT_FALSE(first.deactivate_and_notify);

  // 아직 모터가 안 도달했다고 보고하면(속도 남아있음) 계속 대기.
  for (const std::string& id : {std::string("0"), std::string("2"), std::string("4"),
                                 std::string("22")}) {
    fsm->set_motor_position(id, 0.5);
    fsm->set_motor_velocity(id, 50.0);
  }
  const FsmAction still_moving = fsm->solve_tick();
  EXPECT_FALSE(still_moving.deactivate_and_notify);

  // 전부 도달(위치 0 근처 + 속도 0)했다고 보고하면 그제서야 deactivate.
  for (const std::string& id : {std::string("0"), std::string("2"), std::string("4"),
                                 std::string("22")}) {
    fsm->set_motor_position(id, 0.0);
    fsm->set_motor_velocity(id, 0.0);
  }
  const FsmAction settled = fsm->solve_tick();
  EXPECT_TRUE(settled.deactivate_and_notify);
}

}  // namespace
