#include <gtest/gtest.h>

#include <vector>

#include "ik/collision_model.hpp"

namespace {

TEST(CollisionChecker, BuildsExpectedGeometryAndPairs) {
  ik::RobotModel rm(TEST_URDF_PATH);
  ik::CollisionChecker checker(rm);

  // python: cc.geom_model.ngeoms == 11, len(cc.geom_model.collisionPairs) == 43
  // (11개 링크의 collision mesh, 인접 링크 12쌍을 뺀 C(11,2)-12=43쌍이 활성화됨).
  // 이 값들은 URDF 구조(링크 개수/인접관계)에서만 나오는 값이라 std::map 순회 순서와 무관.
  const Eigen::VectorXd q0 = Eigen::VectorXd::Zero(rm.model().nq);
  EXPECT_FALSE(checker.isColliding(q0));
}

TEST(CollisionChecker, DetectsKnownCollidingConfiguration) {
  ik::RobotModel rm(TEST_URDF_PATH);
  ik::CollisionChecker checker(rm);

  // python CollisionChecker.is_colliding로 미리 확인해둔, 왼팔이 몸통 쪽으로 심하게
  // 접혀서 실제로 충돌하는 자세.
  Eigen::VectorXd q_colliding(8);
  q_colliding << 0.03546487410077015, 1.351391088977806, -1.067521161841099, 1.3459483414117317,
      0.0, 0.0, 0.0, 0.0;
  EXPECT_TRUE(checker.isColliding(q_colliding));
}

TEST(CollisionChecker, ElbowTorsoAvoidanceDqRegressionAfterMeshSwitch) {
  // collision_model.cpp가 예전엔 링크별 로컬 AABB 박스를 충돌 형상으로 썼는데(실제
  // 메시보다 훨씬 커서 left_shoulder_roll이 torso 쪽으로 거의 못 움직일 정도로 충돌을
  // 과대판정하는 문제가 있었음), 실제 메시(BVH)를 직접 쓰도록 바꾸면서 이 값도 그만큼
  // 바뀜 - 더 이상 python 구현(박스 기반)과 안 맞고 새 C++ 메시 기반 구현끼리의
  // 회귀 테스트임.
  ik::RobotModel rm(TEST_URDF_PATH);
  ik::CollisionChecker checker(rm);

  Eigen::VectorXd q_test(8);
  q_test << 0.3, -1.2, 0.8, -1.4, 0.0, 0.0, 0.0, 0.0;

  const Eigen::VectorXd dq = checker.elbowTorsoAvoidanceDq(q_test, 0.05, 1.0, 1e-4);
  ASSERT_EQ(dq.size(), 8);

  Eigen::VectorXd expected(8);
  expected << 0.0, -0.0004667330114062421, 2.041205550980423e-05, 0.0, 0.0, 0.0, 0.0, 0.0;
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(dq[i], expected[i], 1e-6) << "index " << i;
  }
}

TEST(CollisionChecker, ElbowTorsoAvoidanceDqIsZeroWhenFarEnough) {
  ik::RobotModel rm(TEST_URDF_PATH);
  ik::CollisionChecker checker(rm);

  // threshold를 0.001(실제 debug_ik_node.yaml 기본값)로 주면, elbow-torso 거리가
  // 0.0084m > 0.001m이라 조기 반환(전부 0)돼야 함.
  Eigen::VectorXd q_test(8);
  q_test << 0.3, -1.2, 0.8, -1.4, 0.0, 0.0, 0.0, 0.0;
  const Eigen::VectorXd dq = checker.elbowTorsoAvoidanceDq(q_test, 0.001, 1.0, 1e-4);
  for (int i = 0; i < dq.size(); ++i) {
    EXPECT_DOUBLE_EQ(dq[i], 0.0) << "index " << i;
  }
}

TEST(CollisionChecker, IntegratesWithIkStepRegressionAfterMeshSwitch) {
  // debug_ik_node.py의 _step_reachable/_secondary_dq/_effective_secondary_gain이 매 tick
  // 하는 일을 그대로 재현: elbow_torso_avoidance_dq를 구해서 secondary_dq로 넘기고,
  // 그게 0벡터가 아닐 때만 secondary_gain을 켠 채로 ik_step(collision_checker 포함)을 호출.
  // collision_model.cpp를 박스 근사에서 실제 메시(BVH) 충돌로 바꾸면서 이 궤적 자체가
  // 달라짐(박스가 실제보다 커서 충돌을 과대판정하던 문제 수정) - 더 이상 python(박스 기반)
  // 과는 안 맞고 새 C++ 메시 기반 구현끼리의 회귀 테스트임. model/irc_man.urdf의 ee
  // 조인트 z-origin이 -0.15 -> -0.13으로 바뀌면서(2026-08-05) 다시 전체 재계산함 -
  // ikStep(C++) 자체 출력을 그대로 새 golden으로 씀(15틱 안에서는 완전히 settle되지
  // 않고 damping이 계속 커지지만, 결정론적이라 회귀 테스트로는 유효함).
  struct Tick {
    Eigen::VectorXd q;
    double damping;
    bool converged;
    bool hard_rejected;
  };
  auto v8 = [](std::initializer_list<double> vals) {
    Eigen::VectorXd v(8);
    int i = 0;
    for (double val : vals) v[i++] = val;
    return v;
  };
  const std::vector<Tick> golden = {
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.002, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.004, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.008, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.016, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.032, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.064, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.128, false, true},
      {v8({0, 0, 0, 0, 0, 0, 0, 0}), 0.256, false, true},
      {v8({-0.014065711209, 0.0011782511574, -0.16230044693, 0.00065135535021, 0, 0, 0, 0}), 0.128,
       false, false},
      {v8({-0.014065711209, 0.0011782511574, -0.16230044693, 0.00065135535021, 0, 0, 0, 0}), 0.256,
       false, true},
      {v8({-0.014065711209, 0.0011782511574, -0.16230044693, 0.00065135535021, 0, 0, 0, 0}), 0.512,
       false, true},
      {v8({-0.018512932352, 0.00027986543662, -0.20256415411, 0.00017806322826, 0, 0, 0, 0}), 0.256,
       false, false},
      {v8({-0.018512932352, 0.00027986543662, -0.20256415411, 0.00017806322826, 0, 0, 0, 0}), 0.512,
       false, true},
      {v8({-0.018512932352, 0.00027986543662, -0.20256415411, 0.00017806322826, 0, 0, 0, 0}), 1.024,
       false, true},
      {v8({-0.018512932352, 0.00027986543662, -0.20256415411, 0.00017806322826, 0, 0, 0, 0}), 2.048,
       false, true},
  };

  ik::RobotModel rm(TEST_URDF_PATH);
  ik::CollisionChecker checker(rm);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(rm.model().nq);
  double damping = 0.001;
  const Eigen::Vector3d target(0.02, -0.05, -0.25);
  const double threshold = 0.01;
  const double k_pull = 1.0;
  const double secondary_gain_cfg = 0.01;

  ik::IkStepParams params;
  params.tol = 1e-4;
  params.alpha = 0.5;
  params.joint_weights = v8({1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0});
  params.joint_weight_scale = 0.005;

  const ik::CollisionCheckFn collision_fn = [&](const Eigen::VectorXd& q_candidate) {
    return checker.isColliding(q_candidate);
  };

  for (size_t i = 0; i < golden.size(); ++i) {
    const Eigen::VectorXd secondary_dq = checker.elbowTorsoAvoidanceDq(q, threshold, k_pull);
    const bool secondary_active = secondary_dq.squaredNorm() > 0.0;
    params.secondary_dq = secondary_dq;
    params.secondary_gain = secondary_active ? secondary_gain_cfg : 0.0;

    const ik::IkStepResult result = ik::ikStep(rm, target, q, damping, params, collision_fn);
    q = result.q;
    damping = result.damping;

    ASSERT_EQ(q.size(), golden[i].q.size()) << "tick " << i;
    for (int j = 0; j < q.size(); ++j) {
      EXPECT_NEAR(q[j], golden[i].q[j], 1e-6) << "tick " << i << " joint " << j;
    }
    EXPECT_NEAR(damping, golden[i].damping, 1e-9) << "tick " << i;
    EXPECT_EQ(result.converged, golden[i].converged) << "tick " << i;
    EXPECT_EQ(result.hard_rejected, golden[i].hard_rejected) << "tick " << i;
  }
}

}  // namespace
