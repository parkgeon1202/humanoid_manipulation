#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

// 손 끝 목표(target_pos_)를 시간에 따라 미리 그려둔 경로를 따라 이동시키기 위한
// 경유점 보간 모듈 - place_trajectory_(VIRTUAL_PLACE/COMPLETE_PLACE)와
// pick_approach_trajectory_(PICK) 둘 다 이 TrajectoryGenerator를 씀(place 전용이
// 아니라서 이름에 "place"를 안 넣음). IKwalk의 walk_pattern.hpp(foot_trajectory) +
// kick_module.hpp(kick_trajectory) 패턴을 그대로 포팅함 - 경유점(time, pos)을 put_point로
// 순서대로 추가하면 인접한 두 점 사이를 5차 다항식으로 이어 붙여 위치/속도/가속도가
// 연속인 궤적을 만든다. 각 점의 속도는 더 이상 호출부가 손으로 찍지 않고, IPTP식으로
// 자동 추정함(estimateVelocities() 참고) - 시작/끝점은 정지(0), 내부점은 이웃 구간의
// 평균 기울기.
namespace ik
{

// IPTP(Iterative Parabolic Time Parameterization)식 waypoint 속도 추정 - 위치만 정해진
// waypoint 시퀀스에서 각 내부 점의 속도를 이웃 두 구간의 평균 기울기(중심차분)로 채워서
// 돌려줌. 이웃 두 구간의 부호가 반대면(그 점에서 방향이 꺾이는 로컬 극값) 속도를 0으로
// 강제해서 quintic이 그 지점을 오버슈트하지 않게 함. 양 끝점 속도는 호출부가 그대로
// 지정(Trajectory1D::computeSegments()는 항상 0/0을 넘김 - 경로 시작/끝은 정지 상태).
std::vector<double> estimateVelocities(
  const std::vector<double> & times, const std::vector<double> & positions,
  double v_start, double v_end);

class Trajectory1D
{
public:
  void put_point(double time, double pos);
  void clear();

  // 경유점이 하나뿐이면 그 값을 그대로 돌려주고(대기), t가 등록된 시간 범위를
  // 벗어나면 양 끝 경유점 값으로 고정함(walk_pattern.hpp의 foot_trajectory와 달리
  // duration 이후에도 계속 호출되는 걸 전제 - IK가 수렴할 때까지 매 틱 result()를
  // 호출하므로 범위 밖 t에서도 값이 안정적이어야 함).
  double result(double t) const;

private:
  struct DataPoint
  {
    double time;
    double pos;
  };
  // fifthOrderSegment() 경계 조건 - vel은 computeSegments()가 estimateVelocities()로
  // 채우고, acc는 아무 waypoint도 안 쓰므로 항상 0.
  struct BoundaryCondition
  {
    double time;
    double pos;
    double vel;
    double acc;
  };
  struct Segment
  {
    std::array<double, 6> coefs;
    double time_min;
    double time_max;
  };

  std::vector<DataPoint> points_;
  std::vector<Segment> segments_;

  void computeSegments();
  static Segment fifthOrderSegment(const BoundaryCondition & f, const BoundaryCondition & f1);
};

// x, y, z 3축을 각각 독립된 Trajectory1D로 관리하는 손 끝 목표 궤적.
class TrajectoryGenerator
{
public:
  void put_point(double time, double x, double y, double z);
  void clear();
  Eigen::Vector3d result(double t) const;

private:
  Trajectory1D traj_x_;
  Trajectory1D traj_y_;
  Trajectory1D traj_z_;
};

}  // namespace ik
