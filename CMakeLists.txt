cmake_minimum_required(VERSION 2.8.7)
project(RMBP)

find_package(PythonLibs ${PY_VERSION} REQUIRED)
find_package(Boost 1.58.0 REQUIRED COMPONENTS system python)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")

# include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})


add_subdirectory(lemon)
add_subdirectory(ann)

add_executable(RMBP src/kdtree.cpp src/main.cpp src/pointcloudpair.cpp src/pointmatchgraph.cpp)
target_link_libraries(RMBP lemon)
target_link_libraries(RMBP ann)

# create the lib
add_library(py_rmbp SHARED src/rmbp_python_wrapper.cpp src/kdtree.cpp src/pointcloudpair.cpp src/pointmatchgraph.cpp src/rmbp_python_wrapper.cpp)

# link
target_link_libraries(py_rmbp ${Boost_LIBRARIES})
target_link_libraries(py_rmbp lemon)
target_link_libraries(py_rmbp ann)
