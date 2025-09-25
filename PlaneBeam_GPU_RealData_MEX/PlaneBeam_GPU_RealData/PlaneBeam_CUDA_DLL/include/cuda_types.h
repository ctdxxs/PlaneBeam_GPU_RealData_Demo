/*******************************************************************************
* cuda_types.h
* 
* Overview:
*   This header file defines common data types, constants, and utility macros
*   used in the CUDA-accelerated planar beam processing library. It provides
*   basic type definitions and CUDA error handling mechanisms.
* 
* Main Features:
*   - Defines complex number structure for signal processing
*   - Declares constants for image processing parameter limits
*   - Provides CUDA error checking macros and memory management utilities
* 
* Author:
*   CTDXXS
*   Email: wuwentao@mail.ioa.ac.cn
* 
* Creation Date:
*   September 15, 2024
* 
* Version:
*   V1.0.1
* 
* Dependencies:
*   - CUDA Runtime Library
*   - cuComplex.h
* 
* Copyright:
*   Copyright (c) 2024 CTDXXS
*******************************************************************************/
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <stdio.h>

// Forward declaration to avoid circular includes
typedef struct {
    float real;
    float imag;
} ComplexFloat;

// ===================== Constants and Macro Definitions =====================
#define M_PI                3.14159265358979323846

// Image processing parameter limits
#define X_Pixel_MAX    1000   // Maximum number of pixels in X direction
#define Z_Pixel_MAX    1000   // Maximum number of pixels in Z direction
#define Angle_Num_MAX  64     // Maximum number of plane angles
#define Pitch_Num_MAX  64     // Maximum number of probe elements

// Debug control
#define ENABLE_DEBUG_PRINT 0  // Enable debug printing

// ===================== CUDA Error Checking Macros =====================
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

// Note: Complex type conversion functions have been removed as they are not used in the code
