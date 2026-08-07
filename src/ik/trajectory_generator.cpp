#include "ik/trajectory_generator.hpp"

#include <algorithm>

namespace ik
{

void Trajectory1D::put_point(double time, double pos, double vel, double acc)
{
  points_.push_back({time, pos, vel, acc});
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
  for (size_t i = 0; i + 1 < points_.size(); ++i) {
    segments_.push_back(fifthOrderSegment(points_[i], points_[i + 1]));
  }
}

Trajectory1D::Segment Trajectory1D::fifthOrderSegment(const DataPoint & f, const DataPoint & f1)
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

void TrajectoryGenerator::put_point(double time, double x, double y, double z,
                                 double vx, double vy, double vz,
                                 double ax, double ay, double az)
{
  traj_x_.put_point(time, x, vx, ax);
  traj_y_.put_point(time, y, vy, ay);
  traj_z_.put_point(time, z, vz, az);
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
