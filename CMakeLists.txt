cmake_minimum_required(VERSION 3.8)
project(humanoid_manipulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(dynamixel_hardware_msgs REQUIRED)
find_package(irc_humanoid_interfaces REQUIRED)

# --- C++ IK 라이브러리(include/ik, src/ik) 의존성: lib/ik_solver.py + lib/collision_model.py 포팅용 ---
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(coal REQUIRED)
find_package(ament_index_cpp REQUIRED)
# collision_model.cpp가 URDF의 <collision><mesh>를 파싱하는 데 씀(urdfdom의 C++ 파서,
# pinocchio가 이미 transitively 의존하므로 새 시스템 의존성 추가 없음).
find_package(urdf REQUIRED)

# --- C++ IK 라이브러리: scripts/lib/ik_solver.py + scripts/lib/collision_model.py 포팅.
# rclcpp 의존성 없는 순수 라이브러리라 나중에 ik_solve_node(C++)에서도 그대로 재사용함.
add_library(ik_lib
  src/ik/ik_solver.cpp
  src/ik/collision_model.cpp
  src/ik/place_trajectory.cpp
)
target_include_directories(ik_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(ik_lib PUBLIC
  Eigen3::Eigen
  pinocchio::pinocchio
  ament_index_cpp::ament_index_cpp
)
ament_target_dependencies(ik_lib PUBLIC urdf)

# --- C++ 노드: FSM + IK(ik_lib) + 모터 I/O를 전부 담당(구 ik_solve_node.py 흡수) ---
add_executable(main_node src/main_node.cpp src/fsm.cpp)
target_include_directories(main_node PRIVATE include)
target_link_libraries(main_node ik_lib)
ament_target_dependencies(main_node
  rclcpp
  rclcpp_lifecycle
  std_msgs
  sensor_msgs
  geometry_msgs
  dynamixel_hardware_msgs
  irc_humanoid_interfaces
)

# --- C++ 디버그 노드: main_node.cpp와 동일한 로직, activate_cmd 없이 바로 activate
# 하고 비전을 /vision(Vision2MasterMsg)에서 직접 받음(master 경유 없이 단독 테스트용).
add_executable(debug_node src/debug_node.cpp src/fsm.cpp)
target_include_directories(debug_node PRIVATE include)
target_link_libraries(debug_node ik_lib)
ament_target_dependencies(debug_node
  rclcpp
  rclcpp_lifecycle
  std_msgs
  sensor_msgs
  dynamixel_hardware_msgs
  irc_humanoid_interfaces
)

# --- C++ 디버그 IK 노드: scripts/debug_ik_node.py 포팅. ik_lib(ik_step + CollisionChecker)로
# IK를 계산하고, debug_node.cpp와 같은 방식으로 /dynamixel_control에 직접 publish함.
add_executable(debug_ik_node src/debug_ik_node.cpp)
target_include_directories(debug_ik_node PRIVATE include)
target_link_libraries(debug_ik_node ik_lib)
ament_target_dependencies(debug_ik_node
  rclcpp
  dynamixel_hardware_msgs
  irc_humanoid_interfaces
  sensor_msgs
  visualization_msgs
  ament_index_cpp
)

install(TARGETS main_node debug_node debug_ik_node
  DESTINATION lib/${PROJECT_NAME}
)

# ik_lib는 공유 라이브러리로 빌드되므로(ament의 기본 BUILD_SHARED_LIBS=ON) 실행파일이
# 런타임에 찾을 수 있게 표준 lib/ 경로에 설치해야 함(ament_package()가 여기를
# LD_LIBRARY_PATH에 자동으로 추가해줌).
install(TARGETS ik_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch config model
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ik_solver test/test_ik_solver.cpp)
  target_link_libraries(test_ik_solver ik_lib)
  ament_add_gtest(test_collision_model test/test_collision_model.cpp)
  target_link_libraries(test_collision_model ik_lib)

  # fsm.cpp는 ik_lib에 속하지 않고(main_node/debug_node가 각자 컴파일해 넣는 소스)
  # ik_lib 헤더에만 의존하므로, 테스트도 fsm.cpp를 직접 소스로 넣고 ik_lib에 링크함.
  ament_add_gtest(test_fsm test/test_fsm.cpp src/fsm.cpp)
  target_include_directories(test_fsm PRIVATE include)
  target_link_libraries(test_fsm ik_lib)

  # 테스트는 colcon install 상태(ament_index_cpp의 share 디렉터리 탐색)에 의존하지 않고
  # 소스 트리의 URDF를 직접 가리키게 함 - 노드(debug_ik_node.cpp)는 실제 런타임에
  # ament_index_cpp로 따로 경로를 구함(테스트와 무관).
  target_compile_definitions(test_ik_solver PRIVATE
    TEST_URDF_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model/irc_man.urdf")
  target_compile_definitions(test_collision_model PRIVATE
    TEST_URDF_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model/irc_man.urdf")
  target_compile_definitions(test_fsm PRIVATE
    TEST_URDF_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model/irc_man.urdf")
endif()

ament_package()
