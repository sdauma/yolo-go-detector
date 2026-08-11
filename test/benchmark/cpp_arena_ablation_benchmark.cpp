/**
 * C++ ONNX Runtime C-API Arena / Execution-Mode Ablation Benchmark (v2)
 * Purpose: Resolve the §4.7 C-API latency anomaly — C API@intra_op=12 (771 ms) was
 *          ~31% slower than Python@intra_op=12 (588 ms). Hypothesis: the gap stems
 *          from the harness forcing inter_op=1 (disabling inter-op node parallelism)
 *          while Python/Go baselines use binding-default inter_op>1. This v2 sweep
 *          varies (arena, exec_mode, inter_op, intra_op) to (a) reproduce the original
 *          baseline, (b) test inter_op=default at intra_op=12 against Python@12=588 ms,
 *          (c) match the Go ~6-thread baseline, and (d) keep the Arena ablation.
 *          Results are written to cpp_arena_ablation_v2_*.json (originals preserved).
 *
 * Build (MSVC): see build_cpp_arena.bat  (uses /O2 Release; OpenMP not required,
 *        ORT uses its own intra-op thread pool)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <fstream>
#include <cmath>
#include <string>

#include <windows.h>
#include <psapi.h>

// MinGW sal.h vs onnxruntime_c_api.h SAL macro conflict -- save and undef early
#ifdef __GNUC__
#pragma push_macro("_Check_return_")
#pragma push_macro("_In_reads_")
#pragma push_macro("_In_reads_opt_")
#pragma push_macro("_Inout_updates_")
#pragma push_macro("_Out_writes_")
#pragma push_macro("_Out_writes_opt_")
#pragma push_macro("_Inout_updates_all_")
#pragma push_macro("_Out_writes_bytes_all_")
#pragma push_macro("_Out_writes_all_")
#pragma push_macro("_Success_")
#pragma push_macro("_Outptr_result_buffer_maybenull_")
#undef _Check_return_
#undef _In_reads_
#undef _In_reads_opt_
#undef _Inout_updates_
#undef _Out_writes_
#undef _Out_writes_opt_
#undef _Inout_updates_all_
#undef _Out_writes_bytes_all_
#undef _Out_writes_all_
#undef _Success_
#undef _Outptr_result_buffer_maybenull_
#endif

#ifdef _MSC_VER
#include <onnxruntime_cxx_api.h>
#pragma comment(lib, "onnxruntime.lib")
#else
#ifdef _WIN32
#undef _WIN32
#include <onnxruntime_c_api.h>
#define _WIN32
#else
#include <onnxruntime_c_api.h>
#endif
#endif

#ifdef __GNUC__
#pragma pop_macro("_Check_return_")
#pragma pop_macro("_In_reads_")
#pragma pop_macro("_In_reads_opt_")
#pragma pop_macro("_Inout_updates_")
#pragma pop_macro("_Out_writes_")
#pragma pop_macro("_Out_writes_opt_")
#pragma pop_macro("_Inout_updates_all_")
#pragma pop_macro("_Out_writes_bytes_all_")
#pragma pop_macro("_Out_writes_all_")
#pragma pop_macro("_Success_")
#pragma pop_macro("_Outptr_result_buffer_maybenull_")
#endif

static const OrtApi* g_ort = nullptr;
#define CHECK_STATUS(expr, msg) do { \
    OrtStatus* status = (expr); \
    if (status != nullptr) { \
        const char* errMsg = g_ort->GetErrorMessage(status); \
        std::cerr << msg << ": " << errMsg << std::endl; \
        g_ort->ReleaseStatus(status); \
        return 1; \
    } \
} while(0)

// Process memory (MB): RSS = WorkingSetSize, PM = PrivateUsage (PrivateMemorySize64)
void GetProcessMemMB(double& rssMB, double& pmMB) {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        rssMB = pmc.WorkingSetSize / (1024.0 * 1024.0);
        pmMB = pmc.PrivateUsage / (1024.0 * 1024.0);
    } else {
        rssMB = 0; pmMB = 0;
    }
}

std::vector<float> LoadInputData(const std::string& path, size_t expected_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open input data file: " << path << std::endl; return {}; }
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    if (fileSize < expected_size * sizeof(float)) {
        std::cerr << "Input data file size insufficient" << std::endl; return {};
    }
    std::vector<float> data(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    return data;
}

struct ConfigResult {
    std::string arena;
    std::string exec_mode;
    int inter_op;
    int intra_op;
    double avg_latency_ms;
    double p50_latency_ms;
    double p90_latency_ms;
    double throughput_reqs;
    double peak_rss_mb;
    double peak_pm_mb;
};

int RunConfig(bool arenaOn, bool execParallel,
              int intraOp, int interOp, int numInferences,
              const std::wstring& modelPathW,
              const std::vector<float>& inputData,
              ConfigResult& out) {
    OrtEnv* env = nullptr;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cpp_arena_ablation", &env),
                 "createEnv failed");

    OrtSessionOptions* sessionOpts = nullptr;
    CHECK_STATUS(g_ort->CreateSessionOptions(&sessionOpts), "createSessionOptions failed");

    g_ort->SetIntraOpNumThreads(sessionOpts, intraOp);
    g_ort->SetInterOpNumThreads(sessionOpts, interOp);
    g_ort->SetSessionExecutionMode(sessionOpts, execParallel ? ORT_PARALLEL : ORT_SEQUENTIAL);
    g_ort->SetSessionGraphOptimizationLevel(sessionOpts, ORT_ENABLE_ALL);
    if (arenaOn) {
        CHECK_STATUS(g_ort->EnableCpuMemArena(sessionOpts), "EnableCpuMemArena failed");
    } else {
        CHECK_STATUS(g_ort->DisableCpuMemArena(sessionOpts), "DisableCpuMemArena failed");
    }

    OrtSession* session = nullptr;
    CHECK_STATUS(g_ort->CreateSession(env, modelPathW.c_str(), sessionOpts, &session),
                 "createSession failed");

    OrtMemoryInfo* memoryInfo = nullptr;
    CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo),
                 "createMemoryInfo failed");

    const size_t inputSize = 1 * 3 * 640 * 640;
    std::vector<int64_t> inputShapeVec = {1, 3, 640, 640};
    std::vector<int64_t> outputShapeVec = {1, 84, 8400};
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    OrtValue* inputTensor = nullptr;
    CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
        memoryInfo, (void*)inputData.data(), inputSize * sizeof(float),
        inputShapeVec.data(), inputShapeVec.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputTensor), "createTensor failed");

    OrtAllocator* allocator = nullptr;
    CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&allocator), "GetAllocator failed");
    OrtValue* outputTensor = nullptr;
    CHECK_STATUS(g_ort->CreateTensorAsOrtValue(
        allocator, outputShapeVec.data(), outputShapeVec.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputTensor), "createTensor failed");

    // warmup
    for (int i = 0; i < 20; i++) {
        OrtValue* outputs[] = {outputTensor};
        CHECK_STATUS(g_ort->Run(session, nullptr, inputNames,
            (const OrtValue* const*)&inputTensor, 1, outputNames, 1, outputs),
            "warmup failed");
    }

    std::vector<double> latencies;
    latencies.reserve(numInferences);

    double peakRss = 0, peakPm = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numInferences; i++) {
        auto inferStart = std::chrono::high_resolution_clock::now();
        OrtValue* outputs[] = {outputTensor};
        OrtStatus* status = g_ort->Run(session, nullptr, inputNames,
            (const OrtValue* const*)&inputTensor, 1, outputNames, 1, outputs);
        auto inferEnd = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        latencies.push_back(lat);
        if (status != nullptr) {
            std::cerr << "inference #" << i << " failed: "
                      << g_ort->GetErrorMessage(status) << std::endl;
            g_ort->ReleaseStatus(status);
        }
        if (i % 100 == 0) {
            double rss, pm; GetProcessMemMB(rss, pm);
            if (rss > peakRss) peakRss = rss;
            if (pm > peakPm) peakPm = pm;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime - startTime).count();

    std::sort(latencies.begin(), latencies.end());
    double sumLat = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avgLat = sumLat / latencies.size();
    int n = (int)latencies.size();
    double p50 = latencies[n * 50 / 100];
    double p90 = latencies[n * 90 / 100];
    double throughput = n / duration;

    out.arena = arenaOn ? "on" : "off";
    out.exec_mode = execParallel ? "parallel" : "sequential";
    out.inter_op = interOp;
    out.intra_op = intraOp;
    out.avg_latency_ms = avgLat;
    out.p50_latency_ms = p50;
    out.p90_latency_ms = p90;
    out.throughput_reqs = throughput;
    out.peak_rss_mb = peakRss;
    out.peak_pm_mb = peakPm;

    g_ort->ReleaseValue(outputTensor);
    g_ort->ReleaseValue(inputTensor);
    g_ort->ReleaseMemoryInfo(memoryInfo);
    g_ort->ReleaseSessionOptions(sessionOpts);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);
    return 0;
}

int main(int argc, char* argv[]) {
    std::cout << "===== C++ ONNX Runtime C-API Arena/ExecMode Ablation =====" << std::endl;

    const char* basePath = "..\\..";
    std::string resultsDir = std::string(basePath) + "\\results";
    std::string modelName = "yolo11x";
    std::string modelFile = "yolo11x.onnx";
    int numInferences = 1000;

    if (argc > 1) modelName = argv[1];
    if (argc > 2) modelFile = argv[2];
    if (argc > 3) numInferences = std::atoi(argv[3]);

    std::string modelPath = std::string(basePath) + "\\third_party\\" + modelFile;
    std::string inputPath = std::string(basePath) + "\\test\\data\\input_data.bin";
    std::wstring modelPathW = std::wstring(modelPath.begin(), modelPath.end());

    std::cout << "Model: " << modelName << " (" << modelFile << ")" << std::endl;
    std::cout << "Inferences per config: " << numInferences << std::endl;
    std::cout << "graph_opt=ENABLE_ALL (per-config intra_op/inter_op vary, see below)"
              << std::endl << std::endl;

    auto inputData = LoadInputData(inputPath, 1 * 3 * 640 * 640);
    if (inputData.empty()) { std::cerr << "loadInput data failed" << std::endl; return 1; }

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    g_ort = api;

    // Ensure results directory exists BEFORE the sweep loop: per-config JSON files
    // are written inside the loop, so otherwise they would silently fail if absent.
    CreateDirectoryA(resultsDir.c_str(), nullptr);

    // v2 comprehensive sweep to isolate the §4.7 C-API latency anomaly.
    // (arenaOn, execParallel, interOp, intraOp)
    //  - cfg1 reproduces the original baseline (arena on, inter_op=1, intra_op=12 -> ~771ms)
    //  - cfg4 (inter_op=default=0) at intra_op=12 is the DECISIVE test vs Python@12=588ms
    //  - cfg5 (intra_op=6, inter_op=default) matches the Go ~6-thread baseline
    //  - remaining configs keep Arena ON/OFF x exec mode for the §4.6.2(6) ablation
    struct Cfg { bool arenaOn; bool execParallel; int interOp; int intraOp; };
    Cfg configs[] = {
        {true,  false, 1, 12},  // reproduce original cpp_baseline (inter_op=1)
        {true,  false, 2, 12},  // inter_op sensitivity
        {true,  false, 4, 12},  // inter_op sensitivity
        {true,  false, 0, 12},  // inter_op=default (DECISIVE vs Python@12=588)
        {true,  false, 0, 6},   // match Go ~6-thread baseline
        {false, false, 0, 12},  // Arena OFF, decisive config
        {true,  true,  0, 12},  // exec parallel, decisive config
        {false, true,  0, 12},  // Arena OFF + parallel
    };
    const int numConfigs = sizeof(configs) / sizeof(configs[0]);

    std::vector<ConfigResult> results;
    for (int c = 0; c < numConfigs; c++) {
        bool arenaOn = configs[c].arenaOn;
        bool execParallel = configs[c].execParallel;
        int interOp = configs[c].interOp;
        int intraOp = configs[c].intraOp;
        std::cout << "---- Config " << (c + 1) << "/" << numConfigs
                  << ": arena=" << (arenaOn ? "ON" : "OFF")
                  << " exec=" << (execParallel ? "PARALLEL" : "SEQUENTIAL")
                  << " inter_op=" << interOp << " intra_op=" << intraOp << " ----" << std::endl;
        ConfigResult r;
        int rc = RunConfig(arenaOn, execParallel, intraOp, interOp, numInferences,
                           modelPathW, inputData, r);
        if (rc != 0) { std::cerr << "Config " << (c + 1) << " failed" << std::endl; return 1; }

        // DECISIVE TEST (cfg4, c==3): C@intra_op=12, inter_op=default vs Python@12=588.12 ms.
        // Print immediately so the root-cause verdict is known without waiting for all 8 configs.
        if (c == 3) {
            const double python12 = 588.12;
            double delta = (r.avg_latency_ms - python12) / python12 * 100.0;
            std::cout << "  [DECISIVE] C@intra_op=12,inter_op=default = " << r.avg_latency_ms
                      << " ms  vs Python@12=588.12 ms  (delta=" << delta << "%)\n";
            if (std::abs(delta) <= 5.0)
                std::cout << "  [DECISIVE] CONVERGED: anomaly resolved -> inter_op=1 was the root cause.\n";
            else
                std::cout << "  [DECISIVE] residual gap remains -> other factors also contribute.\n";
        }
        std::cout << "  avg=" << r.avg_latency_ms << " ms  p50=" << r.p50_latency_ms
                  << " ms  p90=" << r.p90_latency_ms << " ms  tp=" << r.throughput_reqs
                  << " REQ/s  peakRSS=" << r.peak_rss_mb << " MB  peakPM=" << r.peak_pm_mb
                  << " MB" << std::endl << std::endl;
        results.push_back(r);

        // Persist each config immediately so partial results survive if the run is interrupted
        {
            std::string cfgFile = resultsDir + "\\cpp_arena_ablation_v2_cfg" + std::to_string(c + 1) + ".json";
            std::ofstream cf(cfgFile);
            if (cf.is_open()) {
                cf << "{\n";
                cf << "  \"config_index\": " << (c + 1) << ",\n";
                cf << "  \"arena\": \"" << r.arena << "\",\n";
                cf << "  \"exec_mode\": \"" << r.exec_mode << "\",\n";
                cf << "  \"inter_op\": " << r.inter_op << ",\n";
                cf << "  \"intra_op\": " << r.intra_op << ",\n";
                cf << "  \"avg_latency_ms\": " << r.avg_latency_ms << ",\n";
                cf << "  \"p50_latency_ms\": " << r.p50_latency_ms << ",\n";
                cf << "  \"p90_latency_ms\": " << r.p90_latency_ms << ",\n";
                cf << "  \"throughput_reqs\": " << r.throughput_reqs << ",\n";
                cf << "  \"peak_rss_mb\": " << r.peak_rss_mb << ",\n";
                cf << "  \"peak_pm_mb\": " << r.peak_pm_mb << "\n";
                cf << "}\n";
                cf.close();
            }
        }
    }

    // Save aggregated results (directory already created before the sweep loop)
    std::string resultFile = resultsDir + "\\cpp_arena_ablation_v2_result.json";
    std::ofstream jsonFile(resultFile);
    if (jsonFile.is_open()) {
        jsonFile << "{\n";
        jsonFile << "  \"test_name\": \"C++_Arena_ExecMode_Ablation_v2\",\n";
        jsonFile << "  \"model\": \"" << modelName << "\",\n";
        jsonFile << "  \"graph_optimization_level\": \"ORT_ENABLE_ALL\",\n";
        jsonFile << "  \"inferences_per_config\": " << numInferences << ",\n";
        jsonFile << "  \"configs\": [\n";
        for (size_t i = 0; i < results.size(); i++) {
            const ConfigResult& r = results[i];
            jsonFile << "    {\n";
            jsonFile << "      \"arena\": \"" << r.arena << "\",\n";
            jsonFile << "      \"exec_mode\": \"" << r.exec_mode << "\",\n";
            jsonFile << "      \"inter_op\": " << r.inter_op << ",\n";
            jsonFile << "      \"intra_op\": " << r.intra_op << ",\n";
            jsonFile << "      \"avg_latency_ms\": " << r.avg_latency_ms << ",\n";
            jsonFile << "      \"p50_latency_ms\": " << r.p50_latency_ms << ",\n";
            jsonFile << "      \"p90_latency_ms\": " << r.p90_latency_ms << ",\n";
            jsonFile << "      \"throughput_reqs\": " << r.throughput_reqs << ",\n";
            jsonFile << "      \"peak_rss_mb\": " << r.peak_rss_mb << ",\n";
            jsonFile << "      \"peak_pm_mb\": " << r.peak_pm_mb << "\n";
            jsonFile << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        jsonFile << "  ]\n";
        jsonFile << "}\n";
        jsonFile.close();
        std::cout << "Results saved to: " << resultFile << std::endl;
    }

    std::cout << "\nC++ arena/execmode ablation completed!" << std::endl;
    return 0;
}
