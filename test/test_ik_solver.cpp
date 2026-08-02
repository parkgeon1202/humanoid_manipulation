#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "ik/ik_solver.hpp"

namespace {

// -------------------- detail::solveBoundedLsq --------------------

TEST(BoundedLsq, MatchesUnconstrainedNormalEquations) {
  // 경계를 아주 넓게 주면(무제약과 동일) 표준 최소자승 해와 일치해야 함.
  Eigen::MatrixXd A(4, 2);
  A << 1, 0, 0, 1, 1, 1, 2, 1;
  Eigen::VectorXd b(4);
  b << 1, 2, 3, 4;
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, -1e6);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(2, 1e6);

  const Eigen::VectorXd x = ik::detail::solveBoundedLsq(A, b, lb, ub);
  const Eigen::VectorXd expected = A.colPivHouseholderQr().solve(b);

  ASSERT_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i], expected[i], 1e-9);
  }
}

TEST(BoundedLsq, ClampsToViolatedBound) {
  // 무제약 최적해가 x=[5,5] 근처인데 ub=[1,1]로 강하게 제약하면 두 변수 다 상한에 붙어야 함.
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd b(2);
  b << 5.0, 5.0;
  Eigen::VectorXd lb(2);
  lb << -10.0, -10.0;
  Eigen::VectorXd ub(2);
  ub << 1.0, 1.0;

  const Eigen::VectorXd x = ik::detail::solveBoundedLsq(A, b, lb, ub);
  EXPECT_NEAR(x[0], 1.0, 1e-9);
  EXPECT_NEAR(x[1], 1.0, 1e-9);
}

TEST(BoundedLsq, FixedIndicesStayNearZero) {
  // ik_step이 fixed_indices에 하는 것과 동일하게 lb=ub=+-1e-9로 묶으면 그 변수는
  // 손실을 줄일 수 있어도 거의 0으로 고정돼야 함.
  Eigen::MatrixXd A(3, 3);
  A << 1, 1, 0, 0, 1, 1, 1, 0, 1;
  Eigen::VectorXd b(3);
  b << 3, 3, 3;
  Eigen::VectorXd lb(3);
  lb << -1e6, -1e-9, -1e6;
  Eigen::VectorXd ub(3);
  ub << 1e6, 1e-9, 1e6;

  const Eigen::VectorXd x = ik::detail::solveBoundedLsq(A, b, lb, ub);
  EXPECT_NEAR(x[1], 0.0, 1e-8);
}

// scipy.optimize.lsq_linear(A, b, bounds=(lb, ub))로 미리 뽑아둔 골든값과 대조
// (np.random.seed(42), 8x5 랜덤 행렬 3개). solveBoundedLsq가 scipy의 'bvls' 방식과
// 동일한 결과를 내는지 확인.
struct RandomLsqCase {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::VectorXd lb;
  Eigen::VectorXd ub;
  Eigen::VectorXd expected_x;
};

RandomLsqCase makeCase0() {
  RandomLsqCase c;
  c.A = Eigen::MatrixXd(8, 5);
  c.A << 0.4967141530112327, -0.13826430117118466, 0.6476885381006925, 1.5230298564080254,
      -0.23415337472333597, -0.23413695694918055, 1.5792128155073915, 0.7674347291529088,
      -0.4694743859349521, 0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
      0.24196227156603412, -1.913280244657798, -1.7249178325130328, -0.5622875292409727,
      -1.0128311203344238, 0.3142473325952739, -0.9080240755212109, -1.4123037013352915,
      1.465648768921554, -0.22577630048653566, 0.06752820468792384, -1.4247481862134568,
      -0.5443827245251827, 0.11092258970986608, -1.1509935774223028, 0.37569801834567196,
      -0.600638689918805, -0.2916937497932768, -0.6017066122293969, 1.8522781845089378,
      -0.013497224737933921, -1.0577109289559004, 0.822544912103189, -1.2208436499710222,
      0.2088635950047554, -1.9596701238797756, -1.3281860488984305, 0.19686123586912352;
  c.b = Eigen::VectorXd(8);
  c.b << 0.7384665799954104, 0.1713682811899705, -0.11564828238824053, -0.3011036955892888,
      -1.4785219903674274, -0.7198442083947086, -0.4606387709597875, 1.0571222262189157;
  c.lb = Eigen::VectorXd(5);
  c.lb << -0.3528410587186427, -0.5884264748424236, -0.2268318024772864, -0.8219772826786357,
      -0.16709557931179375;
  c.ub = Eigen::VectorXd(5);
  c.ub << 0.9881982429404655, 0.7950202923669917, 0.2788441133807552, 0.10496990541124217,
      0.8339152856093507;
  c.expected_x = Eigen::VectorXd(5);
  c.expected_x << -0.35284105871864263, 0.0007049950452733113, -0.22683180246805035,
      0.10496990541124215, 0.10175094138114314;
  return c;
}

RandomLsqCase makeCase1() {
  RandomLsqCase c;
  c.A = Eigen::MatrixXd(8, 5);
  c.A << 1.030999522495951, 0.9312801191161986, -0.8392175232226385, -0.3092123758512146,
      0.33126343140356396, 0.9755451271223592, -0.47917423784528995, -0.18565897666381712,
      -1.1063349740060282, -1.1962066240806708, 0.812525822394198, 1.356240028570823,
      -0.07201012158033385, 1.0035328978920242, 0.36163602504763415, -0.6451197546051243,
      0.36139560550841393, 1.5380365664659692, -0.03582603910995154, 1.5646436558140062,
      -2.6197451040897444, 0.8219025043752238, 0.08704706823817122, -0.29900735046586746,
      0.0917607765355023, -1.9875689146008928, -0.21967188783751193, 0.3571125715117464,
      1.477894044741516, -0.5182702182736474, -0.8084936028931876, -0.5017570435845365,
      0.9154021177020741, 0.32875110965968446, -0.5297602037670388, 0.5132674331133561,
      0.09707754934804039, 0.9686449905328892, -0.7020530938773524, -0.3276621465977682;
  c.b = Eigen::VectorXd(8);
  c.b << -0.39210815313215763, -1.4635149481321186, 0.29612027706457605, 0.26105527217988933,
      0.00511345664246089, -0.23458713337514692, -1.4153707420504142, -0.42064532276535904;
  c.lb = Eigen::VectorXd(5);
  c.lb << -0.20787883060031453, -0.40385365426326514, -0.9486187335212672, -0.3908826388186797,
      -0.5669115595690295;
  c.ub = Eigen::VectorXd(5);
  c.ub << 0.7327170630056601, 0.42726664214136456, 0.9746038744488646, 0.9662025654479001,
      0.3266040662428278;
  c.expected_x = Eigen::VectorXd(5);
  c.expected_x << -0.022771121139578226, 0.1086376981457758, -0.27106716878513026,
      0.2335574878378145, 0.3266040662427015;
  return c;
}

RandomLsqCase makeCase2() {
  RandomLsqCase c;
  c.A = Eigen::MatrixXd(8, 5);
  c.A << -1.9187712152990415, -0.026513875449216878, 0.06023020994102644, 2.463242112485286,
      -0.19236096478112252, 0.30154734233361247, -0.03471176970524331, -1.168678037619532,
      1.1428228145150205, 0.7519330326867741, 0.7910319470430469, -0.9093874547947389,
      1.4027943109360992, -1.4018510627922809, 0.5868570938002703, 2.1904556258099785,
      -0.9905363251306883, -0.5662977296027719, 0.09965136508764122, -0.5034756541161992,
      -1.5506634310661327, 0.06856297480602733, -1.0623037137261049, 0.4735924306351816,
      -0.9194242342338032, 1.5499344050175394, -0.7832532923362371, -0.3220615162056756,
      0.8135172173696698, -1.2308643164339552, 0.22745993460412942, 1.307142754282428,
      -1.6074832345612275, 0.1846338585323042, 0.25988279424842353, 0.7818228717773104,
      -1.236950710878082, -1.3204566130842763, 0.5219415656168976, 0.29698467323318606;
  c.b = Eigen::VectorXd(8);
  c.b << 0.25049285034587654, 0.3464482094969757, -0.6800247215784908, 0.23225369716100355,
      0.29307247329868125, -0.7143514180263678, 1.8657745111447566, 0.4738329209117875;
  c.lb = Eigen::VectorXd(5);
  c.lb << -0.6867651335523405, -0.3018423785145038, -0.7409612992127823, -0.3135241787471201,
      -0.39285972834334093;
  c.ub = Eigen::VectorXd(5);
  c.ub << 0.7718422646062217, 0.6846696091424932, 0.8643010694447602, 0.691851603070309,
      0.6114777430019245;
  c.expected_x = Eigen::VectorXd(5);
  c.expected_x << 0.013866465552701048, 0.37221153188803874, -0.5490235668201394,
      -0.005935956264759318, 0.2823693542108029;
  return c;
}

TEST(BoundedLsq, MatchesScipyLsqLinearRandomCases) {
  const std::vector<RandomLsqCase> cases = {makeCase0(), makeCase1(), makeCase2()};
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& c = cases[i];
    const Eigen::VectorXd x = ik::detail::solveBoundedLsq(c.A, c.b, c.lb, c.ub);
    ASSERT_EQ(x.size(), c.expected_x.size()) << "case " << i;
    for (int j = 0; j < x.size(); ++j) {
      EXPECT_NEAR(x[j], c.expected_x[j], 1e-6) << "case " << i << " index " << j;
    }
  }
}

// -------------------- RobotModel --------------------

TEST(RobotModel, LoadsUrdfAndMatchesPythonEePosition) {
  ik::RobotModel rm(TEST_URDF_PATH);
  EXPECT_EQ(rm.model().nq, 8);
  EXPECT_EQ(rm.model().nv, 8);

  const Eigen::VectorXd q0 = Eigen::VectorXd::Zero(rm.model().nq);
  pinocchio::forwardKinematics(rm.model(), rm.data(), q0);
  pinocchio::updateFramePlacements(rm.model(), rm.data());
  const Eigen::Vector3d ee_pos = rm.data().oMf[rm.eeId()].translation();

  // python: lib.ik_solver.data.oMf[ee_id].translation, q=0
  EXPECT_NEAR(ee_pos.x(), 0.0179, 1e-4);
  EXPECT_NEAR(ee_pos.y(), 0.1223, 1e-4);
  EXPECT_NEAR(ee_pos.z(), -0.20665, 1e-4);
}

// -------------------- ikStep golden trace --------------------

TEST(IkStep, MatchesPythonGoldenTrace) {
  // scripts/lib/ik_solver.ik_step를 target=[0.05,0.15,-0.15], joint_weights=
  // [1,10,10,10,1,1,1,1], joint_weight_scale=0.005, secondary_gain=0으로 12틱 돌려서
  // 뽑아둔 golden trace (collision_checker 없음).
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
  // torso_yaw의 URDF 조인트 한계를 +-1.57rad로 좁히면서(2026-08 하드웨어 테스트로
  // 확인된 실제 기구 한계 반영) lb/ub 박스 제약이 바뀌어 이 궤적 자체가 달라짐 -
  // 새 URDF(model/irc_man.urdf)로 다시 뽑아낸 golden trace.
  const std::vector<Tick> golden = {
      {v8({-0.0287489383, -0.0285404293, 0.0569236587, -0.0217385943, 0, 0, 0, 0}), 0.0005, false,
       false},
      {v8({0.7706255308, -0.2490350422, -0.0046568078, -0.3057078719, 0, 0, 0, 0}), 0.0005, false,
       false},
      {v8({0.9543540545, -0.2806615828, -0.2115082814, -0.3889281403, 0, 0, 0, 0}), 0.00025,
       false, false},
      {v8({0.9925926785, -0.2686452864, -0.2956448611, -0.4007541347, 0, 0, 0, 0}), 0.000125,
       false, false},
      {v8({1.0048132352, -0.2600447590, -0.3334175539, -0.4027952867, 0, 0, 0, 0}), 6.25e-05,
       false, false},
      {v8({1.0094735302, -0.2553056419, -0.3513954801, -0.4030724835, 0, 0, 0, 0}), 3.125e-05,
       false, false},
      {v8({1.0115463868, -0.2528406947, -0.3602414267, -0.4030587721, 0, 0, 0, 0}), 1.5625e-05,
       false, false},
      {v8({1.0125751882, -0.2515825803, -0.3646724817, -0.4030229239, 0, 0, 0, 0}), 7.8125e-06,
       false, false},
      {v8({1.0131174732, -0.2509452883, -0.3669126923, -0.4030016305, 0, 0, 0, 0}), 3.90625e-06,
       false, false},
      {v8({1.0134109748, -0.2506235928, -0.3680506269, -0.4029920888, 0, 0, 0, 0}), 1.953125e-06,
       false, false},
      {v8({1.0135712196, -0.2504614847, -0.3686300345, -0.4029886024, 0, 0, 0, 0}), 1e-06,
       false, false},
      {v8({1.0136587689, -0.2503798668, -0.3689254149, -0.4029877028, 0, 0, 0, 0}), 1e-06,
       false, false},
  };

  ik::RobotModel rm(TEST_URDF_PATH);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(rm.model().nq);
  double damping = 0.001;
  const Eigen::Vector3d target(0.05, 0.15, -0.15);

  ik::IkStepParams params;
  params.tol = 1e-4;
  params.alpha = 0.5;
  params.joint_weights = v8({1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0});
  params.joint_weight_scale = 0.005;
  params.secondary_gain = 0.0;

  for (size_t i = 0; i < golden.size(); ++i) {
    const ik::IkStepResult result = ik::ikStep(rm, target, q, damping, params);
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

TEST(IkStep, MatchesClosedFormLmWhenExtraTermsAreOff) {
  // joint_weight_scale=0, secondary_gain=0(기본값)이면 예전 closed-form LM
  // dq = J^T (J J^T + damping^2 I)^-1 e 와 정확히 같은 스텝을 내야 함(관절 한계에서
  // 멀리 떨어진 q + 아주 작은 목표 이동이라 bound가 비활성인 구간에서 검증).
  ik::RobotModel rm(TEST_URDF_PATH);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(rm.model().nq);
  const double damping = 0.01;

  pinocchio::forwardKinematics(rm.model(), rm.data(), q);
  pinocchio::updateFramePlacements(rm.model(), rm.data());
  const Eigen::Vector3d current_pos = rm.data().oMf[rm.eeId()].translation();
  const Eigen::Vector3d target = current_pos + Eigen::Vector3d(0.001, 0.001, 0.001);
  const Eigen::Vector3d error = target - current_pos;

  Eigen::Matrix<double, 6, Eigen::Dynamic> J6 =
      Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, rm.model().nv);
  pinocchio::computeFrameJacobian(rm.model(), rm.data(), q, rm.eeId(),
                                   pinocchio::LOCAL_WORLD_ALIGNED, J6);
  const Eigen::MatrixXd jacobian_pos = J6.topRows(3);

  const double alpha = 0.5;
  const Eigen::MatrixXd JJt = jacobian_pos * jacobian_pos.transpose();
  const Eigen::Vector3d y = (JJt + damping * damping * Eigen::Matrix3d::Identity()).ldlt().solve(error);
  const Eigen::VectorXd dq_closed_form = alpha * (jacobian_pos.transpose() * y);
  Eigen::VectorXd q_expected(rm.model().nq);
  pinocchio::integrate(rm.model(), q, dq_closed_form, q_expected);

  ik::IkStepParams params;
  params.tol = 1e-6;
  params.alpha = alpha;
  const ik::IkStepResult result = ik::ikStep(rm, target, q, damping, params);

  // 이렇게 작은 목표 이동은 선형근사가 정확해서 반드시 accept(rho>good_ratio)돼야 함 -
  // reject라면 애초에 closed-form과 비교할 대상(q_expected)이 아니게 됨.
  ASSERT_FALSE(result.hard_rejected);
  for (int j = 0; j < result.q.size(); ++j) {
    EXPECT_NEAR(result.q[j], q_expected[j], 1e-9) << "joint " << j;
  }
}

TEST(IkStep, NeverExceedsJointLimits) {
  ik::RobotModel rm(TEST_URDF_PATH);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(rm.model().nq);
  double damping = 0.001;
  const Eigen::Vector3d target(0.05, 0.15, -0.15);

  ik::IkStepParams params;
  params.tol = 1e-4;
  params.alpha = 1.0;  // 공격적인 스텝일수록 bound가 실제로 걸리는지 더 잘 드러남

  for (int i = 0; i < 50; ++i) {
    const ik::IkStepResult result = ik::ikStep(rm, target, q, damping, params);
    q = result.q;
    damping = result.damping;
    for (int j = 0; j < q.size(); ++j) {
      EXPECT_GE(q[j], rm.model().lowerPositionLimit[j] - 1e-9) << "tick " << i << " joint " << j;
      EXPECT_LE(q[j], rm.model().upperPositionLimit[j] + 1e-9) << "tick " << i << " joint " << j;
    }
    if (result.converged) break;
  }
}

TEST(IkStep, HardRejectedOnlyWhenCollisionCheckerSaysColliding) {
  ik::RobotModel rm(TEST_URDF_PATH);
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(rm.model().nq);
  const double damping = 0.01;
  const Eigen::Vector3d target(0.05, 0.15, -0.15);
  ik::IkStepParams params;
  params.tol = 1e-4;
  params.alpha = 0.5;

  const ik::IkStepResult always_colliding =
      ik::ikStep(rm, target, q, damping, params, [](const Eigen::VectorXd&) { return true; });
  EXPECT_TRUE(always_colliding.hard_rejected);
  EXPECT_FALSE(always_colliding.converged);
  for (int j = 0; j < q.size(); ++j) {
    EXPECT_DOUBLE_EQ(always_colliding.q[j], q[j]);  // 충돌이면 q는 그대로여야 함
  }
  EXPECT_DOUBLE_EQ(always_colliding.damping, damping * params.damping_increase_factor);

  const ik::IkStepResult never_colliding =
      ik::ikStep(rm, target, q, damping, params, [](const Eigen::VectorXd&) { return false; });
  EXPECT_FALSE(never_colliding.hard_rejected);
}

TEST(IkStep, SecondaryDqForcesAcceptEvenWithBadRho) {
  // python golden case: 이 (q, target, damping, alpha) 조합은 secondary_dq가 없으면
  // rho<=acceptable_ratio라서 반려됨(q 그대로, damping 증가) - secondary_dq!=0이면
  // rho 판정 자체를 건너뛰고 무조건 채택해야 함(이번 세션에서 고친 phantom damping
  // 버그와 짝을 이루는 rho-exemption 로직 검증).
  ik::RobotModel rm(TEST_URDF_PATH);
  Eigen::VectorXd q(8);
  q << 1.3468310248132962, -0.11986458207271178, 0.7731865359248742, -0.007731913537142976, 0.0,
      0.0, 0.0, 0.0;
  const Eigen::Vector3d target(0.12668572679384993, 0.259235811968027, -0.23104042003145686);
  const double damping = 0.004416473057448767;
  const double alpha = 0.96371196431228;

  ik::IkStepParams reject_params;
  reject_params.tol = 1e-4;
  reject_params.alpha = alpha;
  const ik::IkStepResult rejected = ik::ikStep(rm, target, q, damping, reject_params);
  ASSERT_FALSE(rejected.hard_rejected);
  ASSERT_FALSE(rejected.converged);
  EXPECT_NEAR(rejected.damping, damping * reject_params.damping_increase_factor, 1e-9);
  for (int j = 0; j < q.size(); ++j) {
    EXPECT_DOUBLE_EQ(rejected.q[j], q[j]);
  }

  ik::IkStepParams secondary_params = reject_params;
  secondary_params.secondary_dq = Eigen::VectorXd::Zero(8);
  secondary_params.secondary_dq[1] = 0.3;
  secondary_params.secondary_gain = 0.01;
  const ik::IkStepResult accepted = ik::ikStep(rm, target, q, damping, secondary_params);
  EXPECT_FALSE(accepted.hard_rejected);
  EXPECT_FALSE(accepted.converged);
  EXPECT_NEAR(accepted.damping, damping, 1e-9);  // rho 판정을 건너뛰므로 damping 불변
  bool changed = false;
  for (int j = 0; j < q.size(); ++j) {
    if (std::abs(accepted.q[j] - q[j]) > 1e-9) changed = true;
  }
  EXPECT_TRUE(changed);

  Eigen::VectorXd expected(8);
  expected << 0.7262264189169086, -0.055371226029086326, -0.2529464737185134,
      -0.19928813112534421, 0.0, 0.0, 0.0, 0.0;
  for (int j = 0; j < q.size(); ++j) {
    EXPECT_NEAR(accepted.q[j], expected[j], 1e-6) << "joint " << j;
  }
}

}  // namespace
