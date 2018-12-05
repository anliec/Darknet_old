cmake_minimum_required (VERSION 2.8)
project (Darknet)

find_package(OpenCV REQUIRED)
add_compile_definitions(OPENCV)
add_compile_definitions(GPU)

file(GLOB darknet_src
        "src/*.h"
        "src/*.cpp"
        "src/*.c"
        "src/*.hpp"
        "src/*.cu"
        )

include_directories(${OpenCV_INCLUDE_DIRS} include)
add_executable(Darknet examples/darknet.c examples/detector.c ${darknet_src})
target_link_libraries(Darknet ${OpenCV_LIBS})

add_executable(Video_runner examples/video_runner.c ${darknet_src} examples/encrypt_weights.c)
target_link_libraries(Video_runner ${OpenCV_LIBS})

add_executable(encryptor examples/encrypt_weights.c ${darknet_src})

add_executable(decryptor examples/decrypt_weights.c ${darknet_src})
