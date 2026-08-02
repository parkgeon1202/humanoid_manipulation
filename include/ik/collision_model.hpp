#pragma once

#include <map>
#include <string>

#include <Eigen/Dense>
#include <pinocchio/geometry.hpp>

#include "ik/ik_solver.hpp"

// scripts/lib/collision_model.py의 C++ 포팅. URDF <collision><mesh>가 가리키는 실제
// 메쉬(BVH)를 그대로 self-collision 형상으로 씀 - 예전엔 로컬 AABB로 감싼 Box를
// 대신 썼는데, torso_1/left_shoulder_1/left_elbow_1처럼 가늘고 축에 안 맞는 형상은
// AABB가 실제 부피보다 훨씬 커서 충돌을 과대판정하는 문제가 있었음(left_shoulder_roll이
// torso 쪽으로 거의 못 움직이는 정도).
namespace ik {

class CollisionChecker {
 public:
  // rm은 CollisionChecker보다 오래 살아있어야 함(reference로만 들고 있음, ik_lib 전체가
  // 노드/테스트 스코프 안에서 RobotModel 하나를 공유하는 구조를 그대로 반영).
  explicit CollisionChecker(RobotModel& rm);

  // q에서 활성화된(인접하지 않은 링크 쌍) 충돌 쌍 중 하나라도 겹치면 true.
  bool isColliding(const Eigen::VectorXd& q);

  // left_elbow_1-torso_1 메쉬 사이 거리가 threshold(m)보다 가까워지면 그 거리를 벌리는
  // 방향을 돌려줌(secondary_dq로 바로 쓸 수 있음). threshold보다 멀면 zeros(계산 자체도
  // 안 함). torso_yaw + 왼팔 3개 관절에 대해서만 유한차분으로 gradient를 근사함(오른팔/
  // tilt는 이 쌍의 거리에 영향이 없음).
  Eigen::VectorXd elbowTorsoAvoidanceDq(const Eigen::VectorXd& q, double threshold = 0.03,
                                        double k_pull = 1.0, double eps = 1e-4);

 private:
  double pairDistance(const Eigen::VectorXd& q, pinocchio::PairIndex pair_index);

  // 멤버는 반드시 이 선언 순서대로 초기화됨(생성자 초기화 리스트 순서와 무관) -
  // geom_model_ 생성자가 &geom_ids_를 채워야 하므로 geom_ids_가 geom_model_보다 먼저
  // 와야 함(반대 순서면 geom_ids_가 아직 생성되지 않은 메모리를 채우게 되어 UB).
  RobotModel& rm_;
  std::map<std::string, pinocchio::GeomIndex> geom_ids_;
  pinocchio::GeometryModel geom_model_;
  pinocchio::GeometryData geom_data_;
  // isColliding/elbowTorsoAvoidanceDq 전용 Data. RobotModel::data()(ikStep이 쓰는 것)와는
  // 절대 공유하지 않음 - Python의 `self._data = model.createData()`와 동일한 이유
  // (forwardKinematics 결과를 서로 덮어쓰지 않게 분리).
  pinocchio::Data data_;
  pinocchio::PairIndex elbow_torso_pair_index_;
};

}  // namespace ik
