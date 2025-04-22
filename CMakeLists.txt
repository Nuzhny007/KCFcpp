cmake_minimum_required(VERSION 3.8)
project(test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
FILE(GLOB_RECURSE headerfiles "src/*.h*")

add_executable(KCF ${sourcefiles} ${headerfiles})
target_link_libraries(KCF ${OpenCV_LIBS})




