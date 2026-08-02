#include "ik/collision_model.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include <coal/BVH/BVH_model.h>
#include <coal/mesh_loader/loader.h>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/collision/collision.hpp>
#include <pinocchio/collision/distance.hpp>
#include <urdf/model.h>

namespace ik {

namespace {

const std::string kMeshPackagePrefix = "package://humanoid_manipulation/";

struct LinkCollisionMesh {
  std::string mesh_path;
  Eigen::Vector3d scale;
  Eigen::Vector3d origin_xyz;
};

std::string packageRoot(const std::string& urdf_path) {
  // urdf_path == <PACKAGE_ROOT>/model/irc_man.urdf 이므로 두 번 dirname.
  const std::string model_dir = urdf_path.substr(0, urdf_path.find_last_of('/'));
  return model_dir.substr(0, model_dir.find_last_of('/'));
}

std::string resolveMeshPath(const std::string& mesh_filename, const std::string& package_root) {
  std::string relative = mesh_filename;
  const size_t pos = relative.find(kMeshPackagePrefix);
  if (pos != std::string::npos) {
    relative.erase(pos, kMeshPackagePrefix.size());
  }
  return package_root + "/" + relative;
}

// URDF의 각 <link><collision><geometry><mesh>를 읽어서 link 이름 -> (메쉬 경로, scale,
// origin xyz)로 반환. urdfdom(urdf::Model)으로 파싱함 - pinocchio가 이미 이 라이브러리에
// transitively 의존하므로 새 의존성 추가 없음.
std::map<std::string, LinkCollisionMesh> parseLinkCollisionMeshes(const std::string& urdf_path) {
  urdf::Model urdf_model;
  if (!urdf_model.initFile(urdf_path)) {
    throw std::runtime_error("collision_model: failed to parse URDF at " + urdf_path);
  }
  const std::string package_root = packageRoot(urdf_path);

  std::vector<urdf::LinkSharedPtr> links;
  urdf_model.getLinks(links);

  std::map<std::string, LinkCollisionMesh> result;
  for (const auto& link : links) {
    if (!link->collision || !link->collision->geometry) continue;
    const auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->collision->geometry);
    if (!mesh) continue;

    LinkCollisionMesh entry;
    entry.mesh_path = resolveMeshPath(mesh->filename, package_root);
    entry.scale = Eigen::Vector3d(mesh->scale.x, mesh->scale.y, mesh->scale.z);
    entry.origin_xyz = Eigen::Vector3d(link->collision->origin.position.x,
                                        link->collision->origin.position.y,
                                        link->collision->origin.position.z);
    result[link->name] = entry;
  }
  return result;
}

// 두 프레임의 parentJoint가 서로 부모-자식이면(관절 하나로 바로 이어져 있으면) 항상 맞닿는
// 인접 링크로 보고 충돌 쌍에서 제외함.
bool areAdjacent(const pinocchio::Model& model, const pinocchio::Frame& frame_a,
                  const pinocchio::Frame& frame_b) {
  const pinocchio::JointIndex ja = frame_a.parentJoint;
  const pinocchio::JointIndex jb = frame_b.parentJoint;
  return ja == jb || model.parents[ja] == jb || model.parents[jb] == ja;
}

void addCollisionPairs(const pinocchio::Model& model, pinocchio::GeometryModel& geom_model,
                        const std::map<std::string, pinocchio::GeomIndex>& geom_ids) {
  std::vector<std::string> names;
  names.reserve(geom_ids.size());
  for (const auto& kv : geom_ids) names.push_back(kv.first);

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> collision_map =
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
          static_cast<int>(geom_model.ngeoms), static_cast<int>(geom_model.ngeoms), false);
  for (size_t a = 0; a < names.size(); ++a) {
    const pinocchio::Frame& frame_a = model.frames[model.getFrameId(names[a])];
    for (size_t b = a + 1; b < names.size(); ++b) {
      const pinocchio::Frame& frame_b = model.frames[model.getFrameId(names[b])];
      if (!areAdjacent(model, frame_a, frame_b)) {
        const pinocchio::GeomIndex ia = geom_ids.at(names[a]);
        const pinocchio::GeomIndex ib = geom_ids.at(names[b]);
        collision_map(static_cast<int>(ia), static_cast<int>(ib)) = true;
      }
    }
  }
  geom_model.setCollisionPairs(collision_map);
}

// 링크마다 실제 충돌 메시(BVH)를 그대로 충돌 형상으로 씀 - 예전엔 로컬 AABB로 뭉뚱그린
// 박스를 썼는데, torso_1/left_shoulder_1/left_elbow_1처럼 가늘고 축에 안 맞는 형상은
// AABB가 실제 부피보다 훨씬 커서(정상 자세에서도 elbow-torso 거리가 ~1.5mm로 잡히는
// 원인) left_shoulder_roll이 torso 쪽으로 거의 못 움직이는 정도로 충돌이 과대판정됐음.
pinocchio::GeometryModel buildMeshGeometryModel(RobotModel& rm,
                                                 std::map<std::string, pinocchio::GeomIndex>* geom_ids) {
  const pinocchio::Model& model = rm.model();
  pinocchio::GeometryModel geom_model;
  coal::MeshLoader mesh_loader;
  const auto link_meshes = parseLinkCollisionMeshes(rm.urdfPath());

  for (const auto& kv : link_meshes) {
    const std::string& link_name = kv.first;
    const LinkCollisionMesh& mesh_info = kv.second;
    const pinocchio::FrameIndex frame_id = model.getFrameId(link_name);
    const pinocchio::Frame& frame = model.frames[frame_id];

    coal::BVHModelPtr_t bvh = mesh_loader.load(mesh_info.mesh_path, mesh_info.scale);

    const pinocchio::SE3 mesh_in_frame(Eigen::Matrix3d::Identity(), mesh_info.origin_xyz);
    const pinocchio::SE3 placement = frame.placement * mesh_in_frame;

    pinocchio::GeometryObject geom_obj(link_name + "_mesh", frame.parentJoint, placement, bvh);
    (*geom_ids)[link_name] = geom_model.addGeometryObject(geom_obj);
  }

  addCollisionPairs(model, geom_model, *geom_ids);
  return geom_model;
}

pinocchio::PairIndex findCollisionPairIndex(const pinocchio::GeometryModel& geom_model,
                                             pinocchio::GeomIndex geom_id_a,
                                             pinocchio::GeomIndex geom_id_b) {
  for (size_t i = 0; i < geom_model.collisionPairs.size(); ++i) {
    const auto& pair = geom_model.collisionPairs[i];
    if ((pair.first == geom_id_a && pair.second == geom_id_b) ||
        (pair.first == geom_id_b && pair.second == geom_id_a)) {
      return static_cast<pinocchio::PairIndex>(i);
    }
  }
  throw std::runtime_error("collision pair not registered(인접 링크이거나 이름이 틀림)");
}

}  // namespace

CollisionChecker::CollisionChecker(RobotModel& rm)
    : rm_(rm),
      geom_ids_(),
      geom_model_(buildMeshGeometryModel(rm, &geom_ids_)),
      geom_data_(geom_model_),
      data_(rm.model()),
      elbow_torso_pair_index_(
          findCollisionPairIndex(geom_model_, geom_ids_.at("torso_1"), geom_ids_.at("left_elbow_1"))) {}

bool CollisionChecker::isColliding(const Eigen::VectorXd& q) {
  return pinocchio::computeCollisions(rm_.model(), data_, geom_model_, geom_data_, q, true);
}

double CollisionChecker::pairDistance(const Eigen::VectorXd& q, pinocchio::PairIndex pair_index) {
  pinocchio::forwardKinematics(rm_.model(), data_, q);
  pinocchio::updateGeometryPlacements(rm_.model(), data_, geom_model_, geom_data_, q);
  return pinocchio::computeDistance(geom_model_, geom_data_, pair_index).min_distance;
}

Eigen::VectorXd CollisionChecker::elbowTorsoAvoidanceDq(const Eigen::VectorXd& q, double threshold,
                                                        double k_pull, double eps) {
  const double d0 = pairDistance(q, elbow_torso_pair_index_);
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(rm_.model().nq);
  if (d0 >= threshold) return dq;

  std::vector<int> indices = {kTorsoJointIndex};
  for (int idx : kLeftArmIndices) indices.push_back(idx);

  for (int idx : indices) {
    Eigen::VectorXd q_pert = q;
    q_pert[idx] += eps;
    const double d_pert = pairDistance(q_pert, elbow_torso_pair_index_);
    const double grad_d = (d_pert - d0) / eps;
    dq[idx] = k_pull * 2.0 * (threshold - d0) * grad_d;
  }
  return dq;
}

}  // namespace ik
