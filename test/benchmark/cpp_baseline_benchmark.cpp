/**
 * C++ ONNX Runtime Benchmark Test
 * Purpose: Provide ONNX Runtime C++ API native performance baseline (paper §4.7 / Table 10)
 * 
 * Build Requirements:
 * 1. Install ONNX Runtime SDK (from https://github.com/microsoft/onnxruntime/releases)
 *    Recommended: onnxruntime-win-x64-1.23.2.zip
 * 2. Extract and set ONNXRUNTIME_ROOT environment variable to extraction directory
 * 3. Build with MSVC:
 *    cl /EHsc /O2 /I%ONNXRUNTIME_ROOT%\include cpp_baseline_benchmark.cpp ^
 *       /link %ONNXRUNTIME_ROOT%\lib\onnxruntime.lib
 * 
 * Or use build_cpp.bat script for automatic compilation
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <fstream>
#include <cmath>

#include <windows.h>
#include <psapi.h>

// ---------------------------------------------------------------------------
// External reference baselines (NOT measured by this program)
// Source: paper_final.tex §4.1 / Table 10 (Go & Python ONNX Runtime binding baselines)
// IMPORTANT: These values are quoted from the paper, not produced by this C API test.
//            If the paper's Go/Python baselines are re-measured and change, you MUST
//            update the two constants below to keep this output consistent with the paper.
// ---------------------------------------------------------------------------
static const double kGoBaselineLatencyMs     = 738.48; // YOLO11x, NewSession (≈6 intra-op threads)
static const double kPythonBaselineLatencyMs = 588.12; // YOLO11x, intra_op_num_threads=12

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

// ONNX Runtime C API (use C API only for cross-compiler compatibility)
// MinGW g++ compatibility:in onnxruntime_c_api.h #ifdef _WIN32 sets ORT_API_CALL to _stdcall,
// but MinGW does not support bare _stdcall(requires __stdcall__),causing function pointer declaration parsing failure.
// Solution:temporarily undef _WIN32 to make onnxruntime_c_api.h use non-Windows path,
// include then restore _WIN32.
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

// restore SAL macros
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

// Get process RSS memory(MB)
double GetProcessRSSMB() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), 
                             (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0;
}

// Load binary input data
std::vector<float> LoadInputData(const std::string& path, size_t expected_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open input data file: " << path << std::endl;
        return {};
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize < expected_size * sizeof(float)) {
        std::cerr << "Input data file size insufficient" << std::endl;
        return {};
    }

    std::vector<float> data(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    
    return data;
}

// Check status -- Pure C API method,compatible with MSVC and MinGW
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

int main(int argc, char* argv[]) {
    std::cout << "===== C++ ONNX Runtime Test =====" << std::endl;
    std::cout << "Test Purpose: C++ API baseline (compare with Go/Python)" << std::endl;
    std::cout << std::endl;

    // Path
    const char* basePath = "..\\..";
    
    std::string modelName = "yolo11x";
    std::string modelFile = "yolo11x.onnx";
    int numInferences = 2000;
    int warmupIterations = 20;

    if (argc > 1) modelName = argv[1];
    if (argc > 2) modelFile = argv[2];
    if (argc > 3) numInferences = std::atoi(argv[3]);

    std::string modelPath = std::string(basePath) + "\\third_party\\" + modelFile;
    std::string inputPath = std::string(basePath) + "\\test\\data\\input_data.bin";
    
    // Convert to wide string for ONNX Runtime C API
    std::wstring modelPathW = std::wstring(modelPath.begin(), modelPath.end());

    std::cout << "Model: " << modelName << " (" << modelFile << ")" << std::endl;
    std::cout << "ModelPath: " << modelPath << std::endl;
    std::cout << "Number of inferences: " << numInferences << std::endl;
    std::cout << std::endl;

    // -- loadInput data --
    const int64_t inputShape[] = {1, 3, 640, 640};
    const size_t inputSize = 1 * 3 * 640 * 640;

    auto inputData = LoadInputData(inputPath, inputSize);
    if (inputData.empty()) {
        std::cerr << "loadInput datafailed" << std::endl;
        return 1;
    }
    std::cout << "Input dataload: " << inputData.size() << " elements" << std::endl;

    // -- createONNX Runtime --
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    g_ort = api;
    
    OrtEnv* env = nullptr;
    CHECK_STATUS(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cpp_benchmark", &env),
                 "createfailed");

    OrtSessionOptions* sessionOpts = nullptr;
    CHECK_STATUS(api->CreateSessionOptions(&sessionOpts),
                 "createSessionOptionsfailed");

    // P2:
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
    api->SetIntraOpNumThreads(sessionOpts, 12);
    api->SetInterOpNumThreads(sessionOpts, 1);
    api->SetSessionExecutionMode(sessionOpts, ORT_SEQUENTIAL);
    api->SetSessionGraphOptimizationLevel(sessionOpts, ORT_ENABLE_ALL);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    // createSession
    OrtSession* session = nullptr;
    // Note: undef _WIN32  onnxruntime_c_api.h  Windows Path,
    // ORTCHAR_T = char,Path
    CHECK_STATUS(api->CreateSession(env, modelPathW.c_str(), sessionOpts, &session),
                 "createSessionfailed");

    std::cout << "Sessioncreate" << std::endl;

    // -- Allocate memory --
    OrtMemoryInfo* memoryInfo = nullptr;
    CHECK_STATUS(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo),
                 "createMemoryInfofailed");

    std::vector<int64_t> inputShapeVec = {1, 3, 640, 640};
    std::vector<int64_t> outputShapeVec = {1, 84, 8400};
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    OrtValue* inputTensor = nullptr;
    CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(
        memoryInfo, inputData.data(), inputSize * sizeof(float),
        inputShapeVec.data(), inputShapeVec.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputTensor),
        "createTensorfailed");

    OrtValue* outputTensor = nullptr;
    OrtAllocator* allocator = nullptr;
    CHECK_STATUS(api->GetAllocatorWithDefaultOptions(&allocator),
                 "Failed to get Allocator");
    CHECK_STATUS(api->CreateTensorAsOrtValue(
        allocator,
        outputShapeVec.data(), outputShapeVec.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputTensor),
        "createTensorfailed");

    std::cout << "Tensor allocation completed" << std::endl;

    // -- warmup --
    std::cout << "warmup (" << warmupIterations << " )..." << std::endl;
    for (int i = 0; i < warmupIterations; i++) {
        OrtValue* outputs[] = {outputTensor};
        CHECK_STATUS(api->Run(session, nullptr,
                              inputNames, (const OrtValue* const*)&inputTensor, 1,
                              outputNames, 1, outputs),
                     "Inference failed");
    }

    // -- Test --
    std::cout << "Test (" << numInferences << " inference)..." << std::endl;

    double startRSS = GetProcessRSSMB();
    double peakRSS = startRSS;

    std::vector<double> latencies;
    latencies.reserve(numInferences);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numInferences; i++) {
        auto inferStart = std::chrono::high_resolution_clock::now();

        OrtValue* outputs[] = {outputTensor};
        auto status = api->Run(session, nullptr,
                               inputNames, (const OrtValue* const*)&inputTensor, 1,
                               outputNames, 1, outputs);
        
        auto inferEnd = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        latencies.push_back(lat);

        // Check RSS every 100 iterations
        if (i % 100 == 0) {
            double currentRSS = GetProcessRSSMB();
            if (currentRSS > peakRSS) peakRSS = currentRSS;
        }

        if (status != nullptr) {
            const char* errMsg = api->GetErrorMessage(status);
            std::cerr << "inference #" << i << " failed: " << errMsg << std::endl;
            api->ReleaseStatus(status);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double endRSS = GetProcessRSSMB();
    double duration = std::chrono::duration<double>(endTime - startTime).count();

    // -- Calculate statistics --
    std::sort(latencies.begin(), latencies.end());
    
    double sumLat = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avgLat = sumLat / latencies.size();

    double sumSq = 0.0;
    for (double lat : latencies) {
        double diff = lat - avgLat;
        sumSq += diff * diff;
    }
    double stdLat = std::sqrt(sumSq / latencies.size());

    int n = (int)latencies.size();
    double p50 = latencies[n * 50 / 100];
    double p90 = latencies[n * 90 / 100];
    double p99 = latencies[n * 99 / 100];
    double minLat = latencies.front();
    double maxLat = latencies.back();
    double throughput = n / duration;

    // -- Output results --
    std::cout << std::endl;
    std::cout << "========== Test ==========" << std::endl;
    std::cout << "Model: " << modelName << std::endl;
    std::cout << "Number of inferences: " << n << std::endl;
    std::cout << "Total duration: " << duration << " " << std::endl;
    std::cout << "Throughput: " << throughput << " REQ/s" << std::endl;
    std::cout << "Average latency: " << avgLat << " ms" << std::endl;
    std::cout << "Standard deviation: " << stdLat << " ms" << std::endl;
    std::cout << "P50 latency: " << p50 << " ms" << std::endl;
    std::cout << "P90 latency: " << p90 << " ms" << std::endl;
    std::cout << "P99 latency: " << p99 << " ms" << std::endl;
    std::cout << "Min latency: " << minLat << " ms" << std::endl;
    std::cout << "Max latency: " << maxLat << " ms" << std::endl;
    std::cout << "Start RSS: " << startRSS << " MB" << std::endl;
    std::cout << "Peak RSS: " << peakRSS << " MB" << std::endl;
    std::cout << "End RSS: " << endRSS << " MB" << std::endl;
    std::cout << "RSS drift: " << (endRSS - startRSS) << " MB" << std::endl;

    // -- Cross-reference with Go/Python baseline (paper Table 10) --
    std::cout << std::endl;
    std::cout << "-- Cross-reference with Go/Python baseline (paper Table 10) --" << std::endl;
    std::cout << "Go  (YOLO11x baseline, NewSession): " << kGoBaselineLatencyMs
              << " ms (reference, paper Table 10)" << std::endl;
    std::cout << "Python (YOLO11x baseline, intra_op=12): " << kPythonBaselineLatencyMs
              << " ms (reference, paper Table 10)" << std::endl;
    std::cout << "C API (this test, intra_op=12): " << avgLat << " ms" << std::endl;
    std::cout << "Note: C API latency > Go/Python is an observed anomaly; see paper §4.7 for discussion." << std::endl;


    // -- Save results --
    std::string resultsDir = std::string(basePath) + "\\results";
    std::string resultFile = resultsDir + "\\cpp_baseline_result.json";

    // create
    CreateDirectoryA(resultsDir.c_str(), nullptr);

    std::ofstream jsonFile(resultFile);
    if (jsonFile.is_open()) {
        jsonFile << "{\n";
        jsonFile << "  \"test_name\": \"C++_Baseline_Benchmark\",\n";
        jsonFile << "  \"model\": \"" << modelName << "\",\n";
        jsonFile << "  \"total_inferences\": " << n << ",\n";
        jsonFile << "  \"avg_latency_ms\": " << avgLat << ",\n";
        jsonFile << "  \"std_latency_ms\": " << stdLat << ",\n";
        jsonFile << "  \"p50_latency_ms\": " << p50 << ",\n";
        jsonFile << "  \"p90_latency_ms\": " << p90 << ",\n";
        jsonFile << "  \"p99_latency_ms\": " << p99 << ",\n";
        jsonFile << "  \"min_latency_ms\": " << minLat << ",\n";
        jsonFile << "  \"max_latency_ms\": " << maxLat << ",\n";
        jsonFile << "  \"throughput_reqs\": " << throughput << ",\n";
        jsonFile << "  \"start_rss_mb\": " << startRSS << ",\n";
        jsonFile << "  \"peak_rss_mb\": " << peakRSS << ",\n";
        jsonFile << "  \"end_rss_mb\": " << endRSS << ",\n";
        jsonFile << "  \"rss_drift_mb\": " << (endRSS - startRSS) << ",\n";
        jsonFile << "  \"duration_sec\": " << duration << "\n";
        jsonFile << "}\n";
        jsonFile.close();
        std::cout << "\nResults saved to: " << resultFile << std::endl;
    }

    // -- Cleanup --
    api->ReleaseValue(outputTensor);
    api->ReleaseValue(inputTensor);
    api->ReleaseMemoryInfo(memoryInfo);
    api->ReleaseSessionOptions(sessionOpts);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);

    std::cout << "\nC++ baseline test completed successfully!" << std::endl;
    return 0;
}
