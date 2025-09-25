/*******************************************************************************
* PlaneBeam_CUDA_DLL.h
* 
* Overview:
*   This header file defines the interface for the CUDA-accelerated planar beam
*   processing library. It provides function declarations and data structures
*   for GPU-accelerated ultrasound beamforming and signal processing.
* 
* Main Features:
*   - Defines simulation and beamforming parameter structures
*   - Declares core API functions for initialization, processing, and cleanup
*   - Provides device management and error handling functions
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
*   - cuda_types.h
*   - CUDA Runtime Library
* 
* Copyright:
*   Copyright (c) 2024 CTDXXS
*******************************************************************************/
#pragma once

#include "cuda_types.h"

#ifdef PLANEBEAM_CUDA_DLL_EXPORTS
#define PLANEBEAM_CUDA_API __declspec(dllexport)
#else
#define PLANEBEAM_CUDA_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Simulation parameter structure
typedef struct {
    int Mode;                       // Operation mode (0: initialize, 9: release, etc.)
    double Fs;                      // Sampling frequency
    int    Channels;                // Number of channels
    double ArrayPitch;              // Array element spacing
    double C0;                      // Sound speed
    double T0;                      // Start time
    int    maxLen;                  // Maximum data length per channel
    double F0;                      // Center frequency
    int    BandFiltersCoeSize;      // Bandpass filter coefficient length
    int    DemodFiltersCoeSize;     // Demodulation filter coefficient length
    float  BandFiltersCoe[500];     // Bandpass filter coefficients
    float  DemodFiltersCoe[500];    // Demodulation filter coefficients
} SimPara;

// Beamforming parameter structure
typedef struct {
    float  PlaneAngle[64];          // Plane wave angles
    int    PlaneAngleSize;          // Number of plane wave angles
    int    FrameNum;                // Number of frames
    int    Fnum;                    // Number of array elements
    int    PvectorSize;             // Number of probe elements
    int    XvectorSize;             // Number of X coordinates in imaging region
    int    ZvectorSize;             // Number of Z coordinates in imaging region
    float  Pvector[1000];           // Probe element X coordinates
    float  Xvector[2000];           // Imaging region X coordinates (first row)
    float  Zvector[2000];           // Imaging region Z coordinates (first column)
} BeamPara;

// ===================== Main Interface Functions =====================

/**
 * @brief Initialize CUDA device and memory
 * @param simPara Simulation parameters
 * @param beamPara Beamforming parameters
 * @return 0 for success, non-zero for error
 */
PLANEBEAM_CUDA_API int PlaneBeam_Initialize(const SimPara* simPara, const BeamPara* beamPara);

/**
 * @brief Execute beamforming computation
 * @param srcData Input data (short type)
 * @param outputData Output data (ComplexFloat type)
 * @param dataSize Data size
 * @return 0 for success, non-zero for error
 */
PLANEBEAM_CUDA_API int PlaneBeam_Process(const short* srcData, ComplexFloat* outputData, int dataSize);

/**
 * @brief Release CUDA memory and resources
 * @return 0 for success, non-zero for error
 */
PLANEBEAM_CUDA_API int PlaneBeam_Cleanup(void);

/**
 * @brief Get CUDA device information
 * @param deviceCount Number of devices (output)
 * @param deviceName Device name (output, maximum 256 characters)
 * @return 0 for success, non-zero for error
 */
PLANEBEAM_CUDA_API int PlaneBeam_GetDeviceInfo(int* deviceCount, char* deviceName);

/**
 * @brief Set CUDA device
 * @param deviceId Device ID
 * @return 0 for success, non-zero for error
 */
PLANEBEAM_CUDA_API int PlaneBeam_SetDevice(int deviceId);

/**
 * @brief Get error message
 * @param errorCode Error code
 * @param errorMessage Error message buffer (output)
 * @param bufferSize Buffer size
 * @return 0 for success, non-zero for error
 */
PLANEBEAM_CUDA_API int PlaneBeam_GetErrorMessage(int errorCode, char* errorMessage, int bufferSize);

#ifdef __cplusplus
}
#endif
