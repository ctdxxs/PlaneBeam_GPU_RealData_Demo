#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <stdio.h>

// 前向声明，避免循环包含
typedef struct {
    float real;
    float imag;
} ComplexFloat;

// ===================== 常量与宏定义 =====================
#define M_PI                3.14159265358979323846
#define Filter_Length_Max   300

#define X_Pixel_MAX    1000   // X方向最大像素数
#define Z_Pixel_MAX    1000   // Z方向最大像素数
#define Angle_Num_MAX  64     // 最大平面角度数
#define Pitch_Num_MAX  64     // 最大探头阵元数

// 数学常量
#define eps 1E-6f
#define pi acosf(-1.0)
#define thread_per_block 128

#define ENABLE_DEBUG_PRINT 0  // 启用调试打印

// ===================== CUDA错误检查宏 =====================
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
    }
}
#endif

#define SAFE_CUDA_FREE(ptr) do { if (ptr) { cudaFree(ptr); ptr = nullptr; } } while(0)

// ===================== 复数类型转换 =====================
// 将cuFloatComplex转换为ComplexFloat
inline ComplexFloat cuComplexToComplexFloat(const cuFloatComplex& cu) {
    ComplexFloat cf;
    cf.real = cu.x;
    cf.imag = cu.y;
    return cf;
}

// 将ComplexFloat转换为cuFloatComplex
inline cuFloatComplex complexFloatToCuComplex(const ComplexFloat& cf) {
    return make_cuFloatComplex(cf.real, cf.imag);
}
