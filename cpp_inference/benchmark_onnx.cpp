#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <iomanip>
#include <thread>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <onnx模型路径> [图片尺寸, 默认224] [预热次数, 默认20] [测试次数, 默认100]\n";
        return 1;
    }
    std::string model_path = argv[1];
    int image_size = argc > 2 ? std::stoi(argv[2]) : 224;
    int warmup = argc > 3 ? std::stoi(argv[3]) : 20;
    int runs = argc > 4 ? std::stoi(argv[4]) : 100;

    std::cout << "ℹ️ ONNX模型: " << model_path << "\n";
    std::cout << "📏 输入尺寸: " << image_size << "x" << image_size << "\n";
    std::cout << "🔥 预热次数: " << warmup << "   🚀 测试次数: " << runs << "\n";

    // 1. 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "RepVGG_Benchmark");
    Ort::SessionOptions session_options;
    unsigned int num_threads = std::thread::hardware_concurrency();
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetInterOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    // 2. 获取输入输出名
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    // 3. 构造虚拟输入
    std::vector<int64_t> input_dims = {1, 3, image_size, image_size};
    size_t input_tensor_size = 1 * 3 * image_size * image_size;
    std::vector<float> input_tensor_values(input_tensor_size, 1.0f); // 全1

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_dims.data(), input_dims.size());

    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    // 4. 预热
    std::cout << "🔥 正在预热..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    }

    // 5. 正式测试
    std::cout << "🚀 正在运行基准测试..." << std::endl;
    std::vector<double> timings;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(ms);
    }

    // 6. 统计与打印
    double avg_latency = 0.0;
    for (auto t : timings) avg_latency += t;
    avg_latency /= timings.size();
    double fps = 1000.0 / avg_latency;

    std::cout << "✅ 测试完成!\n";
    std::cout << "⏱️  平均延迟: " << std::fixed << std::setprecision(2) << avg_latency << " ms\n";
    std::cout << "🚀  吞吐量: " << std::fixed << std::setprecision(2) << fps << " FPS\n";
    return 0;
}
