# 指定CMake最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称
project(RepVGG_Inference CXX)

# 设置C++标准为17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 指定 ONNX Runtime 的精确路径 ---
# 根据 `brew --prefix onnxruntime` 的输出设置根路径
set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")

# 设置头文件和库文件的具体路径
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")

# 检查路径是否存在，提供清晰的错误信息
if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR} OR NOT EXISTS ${ONNXRUNTIME_LIBRARY})
    message(FATAL_ERROR "ONNX Runtime paths are incorrect. Please verify the path set in ONNXRUNTIME_ROOT.")
else()
    message(STATUS "✅ Using ONNX Runtime includes: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "✅ Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

# 添加可执行文件目标
add_executable(repvgg_inference main.cpp)

# 将ONNX Runtime的头文件目录添加到目标
target_include_directories(repvgg_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

# 将ONNX Runtime库链接到我们的可执行文件
target_link_libraries(repvgg_inference PRIVATE ${ONNXRUNTIME_LIBRARY})

find_package(OpenCV REQUIRED)
target_link_libraries(repvgg_inference PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS})
target_include_directories(repvgg_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(benchmark_onnx benchmark_onnx.cpp)
target_include_directories(benchmark_onnx PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(benchmark_onnx PRIVATE ${ONNXRUNTIME_LIBRARY}) 