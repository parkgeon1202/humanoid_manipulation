#include <gtest/gtest.h>

#include <cmath>
#include <memory>

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
  tuning.torso_kick_offsets = {0.5, 1.0, 1.5, -0.5, -1.0, -1.5};
  tuning.gripper_open_rad = 0.5;
  tuning.gripper_closed_rad = 0.0;
  tuning.gripper_step_rad = 0.02;
  tuning.torso_target_rad = -1.5;
  tuning.left_shoulder_roll_target_rad = 1.0;
  tuning.settle_velocity_threshold = 0.3;
  tuning.settle_position_tolerance_rad = 0.05;
  tuning.grip_moving_velocity_threshold = 0.3;
  tuning.grip_near_zero_position_threshold = 0.2;
  tuning.grip_wiggle_kick_position_rad = 1.0;

  tuning.home_q_full = Eigen::VectorXd::Zero(8);
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

TEST(ManipulationFsm, PickReadyHomeResetsThenCapturesVisionOnceThenRotatesThenSettles) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);  // SIT -> PICK_READY

  // 첫 tick: 홈 복귀 명령만 나가고 아직 회전 시작 전.
  const FsmAction tick1 = fsm->solve_tick();
  ASSERT_TRUE(tick1.publish_motor_command);
  EXPECT_NEAR(tick1.motor_goal_positions.at("22"), 0.0, 1e-9);
  EXPECT_EQ(tick1.state_string, "PICK.PICK_READY");

  // 비전이 아직 없으면 캡처도 회전도 안 함(publish 여부만 확인 - 내부 상태는
  // solve_tick 반환값의 publish_motor_command로 간접 확인).
  const FsmAction tick2 = fsm->solve_tick();
  EXPECT_FALSE(tick2.publish_motor_command);

  // 비전이 들어오면 그 순간 값을 캡처하고 허리+어깨를 회전시킴.
  fsm->update_vision(0.05, 0.15, -0.15);
  const FsmAction tick3 = fsm->solve_tick();
  ASSERT_TRUE(tick3.publish_motor_command);
  EXPECT_NEAR(tick3.motor_goal_positions.at("22"), -1.5, 1e-9);
  EXPECT_NEAR(tick3.motor_goal_positions.at("2"), 1.0, 1e-9);

  // 아직 모터가 도달했다고 보고하지 않았으면 PICK_READY에 머물러야 함.
  EXPECT_EQ(tick3.state_string, "PICK.PICK_READY");

  // 모터가 목표에 도달(위치 근접 + 속도 0)했다고 보고하면 PICK으로 전이.
  fsm->set_motor_position("22", -1.5);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  const FsmAction tick4 = fsm->solve_tick();
  EXPECT_EQ(tick4.state_string, "PICK.PICK");
}

TEST(ManipulationFsm, PickReadyRotationTargetIgnoresLaterVisionUpdates) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);
  fsm->solve_tick();  // 홈 복귀 tick

  fsm->update_vision(0.05, 0.15, -0.15);
  const FsmAction rotate_tick = fsm->solve_tick();  // 캡처 + 회전 시작
  ASSERT_TRUE(rotate_tick.publish_motor_command);
  EXPECT_NEAR(rotate_tick.motor_goal_positions.at("22"), -1.5, 1e-9);

  // 회전이 이미 시작된 뒤 비전이 계속 바뀌어도(딱 한 번만 캡처하므로) 허리/어깨
  // 목표 각도 자체는 절대 안 바뀌어야 함 - 아직 정착 전이라 매 tick 계속 publish됨.
  fsm->update_vision(-9.0, -9.0, -9.0);
  const FsmAction still_rotating = fsm->solve_tick();
  ASSERT_TRUE(still_rotating.publish_motor_command);
  EXPECT_NEAR(still_rotating.motor_goal_positions.at("22"), -1.5, 1e-9);
  EXPECT_NEAR(still_rotating.motor_goal_positions.at("2"), 1.0, 1e-9);
}

TEST(ManipulationFsm, PickPreservesRotatedQAsIkStartingPoint) {
  auto fsm = makeFsm();
  fsm->on_motion_end(true);
  fsm->solve_tick();  // 홈 복귀
  fsm->update_vision(0.05, 0.15, -0.15);
  fsm->solve_tick();  // 캡처 + 회전 시작

  fsm->set_motor_position("22", -1.5);
  fsm->set_motor_velocity("22", 0.0);
  fsm->set_motor_position("2", 1.0);
  fsm->set_motor_velocity("2", 0.0);
  const FsmAction entered_pick = fsm->solve_tick();
  ASSERT_EQ(entered_pick.state_string, "PICK.PICK");

  // PICK 진입 직후 첫 IK tick: q_가 0으로 리셋되지 않고 torso=-1.5rad에서
  // 시작했으므로, 이번 한 스텝만으로 0 근처로 되돌아갈 리 없음(IK는 alpha=0.5의
  // 점진적 스텝이라 급격한 변화가 없음 - 여전히 -1.0 이하여야 함).
  ASSERT_TRUE(entered_pick.publish_motor_command);
  EXPECT_LT(entered_pick.motor_goal_positions.at("22"), -1.0);
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
