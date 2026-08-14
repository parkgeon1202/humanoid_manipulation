#include "ik/trajectory_generator.hpp"

#include <algorithm>

namespace ik
{

std::vector<double> estimateVelocities(
  const std::vector<double> & times, const std::vector<double> & positions,
  double v_start, double v_end)
{
  const size_t n = times.size();
  std::vector<double> vel(n, 0.0);
  if (n == 0) {
    return vel;
  }
  for (size_t i = 1; i + 1 < n; ++i) {
    const double v_in = (positions[i] - positions[i - 1]) / (times[i] - times[i - 1]);
    const double v_out = (positions[i + 1] - positions[i]) / (times[i + 1] - times[i]);
    vel[i] = (v_in > 0.0) == (v_out > 0.0) ? 0.5 * (v_in + v_out) : 0.0;
  }
  vel.front() = v_start;
  vel.back() = v_end;
  return vel;
}

void Trajectory1D::put_point(double time, double pos)
{
  points_.push_back({time, pos});
  computeSegments();
}

void Trajectory1D::clear()
{
  points_.clear();
  segments_.clear();
}

void Trajectory1D::computeSegments()
{
  segments_.clear();
  if (points_.size() < 2) {
    return;
  }

  std::vector<double> times;
  std::vector<double> positions;
  times.reserve(points_.size());
  positions.reserve(points_.size());
  for (const auto & p : points_) {
    times.push_back(p.time);
    positions.push_back(p.pos);
  }
  // 경로 시작/끝은 정지(0)로 두고, 내부 점들은 IPTP식 중심차분으로 속도 추정
  // (estimateVelocities() 참고).
  const std::vector<double> vel =
    estimateVelocities(times, positions, /*v_start=*/0.0, /*v_end=*/0.0);

  for (size_t i = 0; i + 1 < points_.size(); ++i) {
    segments_.push_back(fifthOrderSegment(
        {times[i], positions[i], vel[i], /*acc=*/0.0},
        {times[i + 1], positions[i + 1], vel[i + 1], /*acc=*/0.0}));
  }
}

Trajectory1D::Segment Trajectory1D::fifthOrderSegment(
  const BoundaryCondition & f, const BoundaryCondition & f1)
{
  const double p0 = f.pos, v0 = f.vel, a0 = f.acc, t0 = f.time;
  const double p1 = f1.pos, v1 = f1.vel, a1 = f1.acc, t1 = f1.time;
  const double T = t1 - t0;

  Segment seg;
  seg.time_min = t0;
  seg.time_max = t1;
  seg.coefs[0] = p0;
  seg.coefs[1] = v0;
  seg.coefs[2] = a0 / 2.0;
  seg.coefs[3] =
    (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T - (3 * a1 - a0) * T * T) / (2 * T * T * T);
  seg.coefs[4] =
    (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T + (3 * a1 - 2 * a0) * T * T) / (2 * T * T * T * T);
  seg.coefs[5] =
    (12 * (p1 - p0) - 6 * T * (v1 + v0) - (a1 - a0) * T * T) / (2 * T * T * T * T * T);
  return seg;
}

double Trajectory1D::result(double t) const
{
  if (segments_.empty()) {
    return points_.empty() ? 0.0 : points_.back().pos;
  }

  const double t_clamped = std::clamp(t, segments_.front().time_min, segments_.back().time_max);
  const Segment * seg = &segments_.back();
  for (const auto & s : segments_) {
    if (t_clamped >= s.time_min && t_clamped <= s.time_max) {
      seg = &s;
      break;
    }
  }

  double pos = 0.0;
  double dt_pow = 1.0;
  const double dt = t_clamped - seg->time_min;
  for (double c : seg->coefs) {
    pos += c * dt_pow;
    dt_pow *= dt;
  }
  return pos;
}

void TrajectoryGenerator::put_point(double time, double x, double y, double z)
{
  traj_x_.put_point(time, x);
  traj_y_.put_point(time, y);
  traj_z_.put_point(time, z);
}

void TrajectoryGenerator::clear()
{
  traj_x_.clear();
  traj_y_.clear();
  traj_z_.clear();
}

Eigen::Vector3d TrajectoryGenerator::result(double t) const
{
  return Eigen::Vector3d(traj_x_.result(t), traj_y_.result(t), traj_z_.result(t));
}

}  // namespace ik
