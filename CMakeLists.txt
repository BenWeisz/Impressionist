cmake_minimum_required(VERSION 2.8)
project( Impressionist )
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Impressionist main.cpp )
target_link_libraries( Impressionist  ${OpenCV_LIBS})
