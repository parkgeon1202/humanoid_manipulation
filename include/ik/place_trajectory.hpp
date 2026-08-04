#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

// place 단계에서 손 끝 목표(target_pos_)를 시간에 따라 미리 그려둔 경로를 따라
// 이동시키기 위한 경유점 보간 모듈. IKwalk의 walk_pattern.hpp(foot_trajectory) +
// kick_module.hpp(kick_trajectory) 패턴을 그대로 포팅함 - 경유점(time, pos, vel, acc)을
// put_point로 순서대로 추가하면 인접한 두 점 사이를 5차 다항식으로 이어 붙여
// 위치/속도/가속도가 연속인 궤적을 만든다.
namespace ik
{

class Trajectory1D
{
public:
  void put_point(double time, double pos, double vel = 0.0, double acc = 0.0);
  void clear();

  // 경유점이 하나뿐이면 그 값을 그대로 돌려주고(대기), t가 등록된 시간 범위를
  // 벗어나면 양 끝 경유점 값으로 고정함(walk_pattern.hpp의 foot_trajectory와 달리
  // duration 이후에도 계속 호출되는 걸 전제 - place는 정해진 duration 없이 IK가
  // 수렴할 때까지 매 틱 result()를 호출하므로 범위 밖 t에서도 값이 안정적이어야 함).
  double result(double t) const;

private:
  struct DataPoint
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
  static Segment fifthOrderSegment(const DataPoint & f, const DataPoint & f1);
};

// x, y, z 3축을 각각 독립된 Trajectory1D로 관리하는 손 끝 목표 궤적.
class PlaceTrajectory
{
public:
  void put_point(double time, double x, double y, double z,
                 double vx = 0.0, double vy = 0.0, double vz = 0.0,
                 double ax = 0.0, double ay = 0.0, double az = 0.0);
  void clear();
  Eigen::Vector3d result(double t) const;

private:
  Trajectory1D traj_x_;
  Trajectory1D traj_y_;
  Trajectory1D traj_z_;
};

}  // namespace ik
