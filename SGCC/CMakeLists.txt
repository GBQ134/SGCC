cmake_minimum_required(VERSION 3.10)
project(SGCC)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)
set(Boost_LIB_PREFIX "lib")
set(Boost_ARCHITECTURE "-x64")
set(OpenCV_DIR "/home/gbq/opencv")

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)


FILE(GLOB_RECURSE HDRS_FILES "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE SRCS_FILES "src/*.c" "src/*.cpp")

# 包含头文件目录

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(SGCC main.cpp ${SRCS_FILES} ${HDRS_FILES})


# 链接OpenCV库
target_link_libraries(SGCC ${PCL_LIBRARIES} ${OpenCV_LIBS})
