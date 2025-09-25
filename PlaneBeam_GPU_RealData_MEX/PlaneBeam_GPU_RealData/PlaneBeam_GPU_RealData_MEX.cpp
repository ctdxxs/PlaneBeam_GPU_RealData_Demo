/*******************************************************************************
* PlaneBeam_GPU_RealData_MEX.cpp
* 
* Overview:
*   This file is a MATLAB MEX interface file that connects MATLAB with CUDA-accelerated
*   planar beam processing library. This interface provides MATLAB with the capability
*   to call GPU-accelerated functions for planar beamforming and data processing.
* 
* Main Features:
*   - Dynamically load PlaneBeam_CUDA_DLL.dll library
*   - Initialize CUDA environment and planar beam parameters
*   - Process real data from MATLAB and return results
*   - Manage CUDA resources and memory release
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
*   - MATLAB MEX Library
*   - CUDA Runtime Library
*   - PlaneBeam_CUDA_DLL.dll
* 
* Copyright:
*   Copyright (c) 2024 CTDXXS
*******************************************************************************/
#include "mex.h"
#include <matrix.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <new>
#include <exception>

// Include dynamic library header files
#include "PlaneBeam_CUDA_DLL/include/PlaneBeam_CUDA_DLL.h"
#include "PlaneBeam_CUDA_DLL/include/cuda_types.h"

// Dynamic library handle
static HMODULE g_hDLL = nullptr;
static bool g_dllLoaded = false;

// Function pointer type definitions
typedef int (*PlaneBeam_Initialize_t)(const SimPara*, const BeamPara*);
typedef int (*PlaneBeam_Process_t)(const short*, ComplexFloat*, int);
typedef int (*PlaneBeam_Cleanup_t)(void);
typedef int (*PlaneBeam_GetDeviceInfo_t)(int*, char*);
typedef int (*PlaneBeam_SetDevice_t)(int);
typedef int (*PlaneBeam_GetErrorMessage_t)(int, char*, int);

// Function pointers
static PlaneBeam_Initialize_t g_PlaneBeam_Initialize = nullptr;
static PlaneBeam_Process_t g_PlaneBeam_Process = nullptr;
static PlaneBeam_Cleanup_t g_PlaneBeam_Cleanup = nullptr;
static PlaneBeam_GetDeviceInfo_t g_PlaneBeam_GetDeviceInfo = nullptr;
static PlaneBeam_SetDevice_t g_PlaneBeam_SetDevice = nullptr;
static PlaneBeam_GetErrorMessage_t g_PlaneBeam_GetErrorMessage = nullptr;

// Global parameter structures
SimPara mSimPara;
BeamPara mBeamPara;

/**
 * @brief Load dynamic library
 * @return true for success, false for failure
 */
bool LoadDLL() {
    if (g_dllLoaded) {
        return true;
    }
    
    // Try multiple possible DLL paths
    const char* dllPaths[] = {
        "PlaneBeam_CUDA_DLL.dll",
        "./PlaneBeam_CUDA_DLL.dll",
        "../bin/x64/Release/PlaneBeam_CUDA_DLL.dll",
        "bin/x64/Release/PlaneBeam_CUDA_DLL.dll"
    };
    
    g_hDLL = nullptr;
    for (int i = 0; i < 4; i++) {
        g_hDLL = LoadLibraryA(dllPaths[i]);
        if (g_hDLL) {
            mexPrintf("Successfully loaded DLL from: %s\n", dllPaths[i]);
            break;
        }
    }
    
    if (!g_hDLL) {
        mexPrintf("Error: Failed to load PlaneBeam_CUDA_DLL.dll from any path\n");
        mexPrintf("Please ensure the DLL is in the MATLAB path or current directory\n");
        return false;
    }
    
    // Get function addresses
    g_PlaneBeam_Initialize = (PlaneBeam_Initialize_t)GetProcAddress(g_hDLL, "PlaneBeam_Initialize");
    g_PlaneBeam_Process = (PlaneBeam_Process_t)GetProcAddress(g_hDLL, "PlaneBeam_Process");
    g_PlaneBeam_Cleanup = (PlaneBeam_Cleanup_t)GetProcAddress(g_hDLL, "PlaneBeam_Cleanup");
    g_PlaneBeam_GetDeviceInfo = (PlaneBeam_GetDeviceInfo_t)GetProcAddress(g_hDLL, "PlaneBeam_GetDeviceInfo");
    g_PlaneBeam_SetDevice = (PlaneBeam_SetDevice_t)GetProcAddress(g_hDLL, "PlaneBeam_SetDevice");
    g_PlaneBeam_GetErrorMessage = (PlaneBeam_GetErrorMessage_t)GetProcAddress(g_hDLL, "PlaneBeam_GetErrorMessage");
    
    if (!g_PlaneBeam_Initialize || !g_PlaneBeam_Process || !g_PlaneBeam_Cleanup) {
        mexPrintf("Error: Failed to get function addresses from DLL\n");
        if (g_hDLL) {
            FreeLibrary(g_hDLL);
            g_hDLL = nullptr;
        }
        return false;
    }
    
    g_dllLoaded = true;
    return true;
}

/**
 * @brief Unload dynamic library
 */
void UnloadDLL() {
    if (g_dllLoaded) {
        try {
            if (g_PlaneBeam_Cleanup) {
                int result = g_PlaneBeam_Cleanup();
                if (result != 0) {
                    mexPrintf("Warning: Cleanup returned error code %d\n", result);
                }
            }
        } catch (...) {
            mexPrintf("Warning: Exception during cleanup\n");
        }
        
        if (g_hDLL) {
            FreeLibrary(g_hDLL);
            g_hDLL = nullptr;
        }
        
        // Reset all function pointers
        g_PlaneBeam_Initialize = nullptr;
        g_PlaneBeam_Process = nullptr;
        g_PlaneBeam_Cleanup = nullptr;
        g_PlaneBeam_GetDeviceInfo = nullptr;
        g_PlaneBeam_SetDevice = nullptr;
        g_PlaneBeam_GetErrorMessage = nullptr;
        
        g_dllLoaded = false;
    }
}

/**
 * @brief Handle CUDA errors
 * @param errorCode Error code
 */
void HandleError(int errorCode) {
    if (errorCode != 0) {
        char errorMessage[256] = {0};
        if (g_PlaneBeam_GetErrorMessage) {
            try {
                if (g_PlaneBeam_GetErrorMessage(errorCode, errorMessage, sizeof(errorMessage)) == 0) {
                    mexErrMsgIdAndTxt("PlaneBeam:Error", "CUDA DLL Error %d: %s", errorCode, errorMessage);
                } else {
                    mexErrMsgIdAndTxt("PlaneBeam:Error", "CUDA DLL Error %d: Unknown error", errorCode);
                }
            } catch (...) {
                mexErrMsgIdAndTxt("PlaneBeam:Error", "CUDA DLL Error %d: Exception during error handling", errorCode);
            }
        } else {
            mexErrMsgIdAndTxt("PlaneBeam:Error", "CUDA DLL Error %d: Error handler not available", errorCode);
        }
    }
}

/**
 * @brief MEX function entry point (MATLAB interface)
 *        Responsible for memory initialization, parameter updates, beamforming, and memory release.
 *        
 *        Supports three operation modes:
 *        - Mode 0: Initialize CUDA device and memory, update simulation and beamforming parameters
 *        - Mode 1: Execute beamforming computation, process input data and output results
 *        - Mode 9: Release CUDA memory and resources
 *
 * @param nlhs Number of left-hand side output parameters (unused)
 * @param plhs Output parameter pointer array (unused)
 * @param nrhs Number of right-hand side input parameters
 * @param prhs Input parameter pointer array
 *              - prhs[0]: Structure containing Mode field
 *              - prhs[1]: Beamforming parameter structure (required for Mode 0)
 *              - prhs[2]: Input data array (required for Mode 1)
 *              - prhs[3]: Output data array (required for Mode 1)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Check input parameter count
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("PlaneBeam:inputError", "At least one input argument required.");
    }
    
    // Load dynamic library
    if (!LoadDLL()) {
        mexErrMsgIdAndTxt("PlaneBeam:loadError", "Failed to load CUDA DLL.");
    }
    
    // Check if first input is a structure
    if (!mxIsStruct(prhs[0])) {
        mexErrMsgIdAndTxt("PlaneBeam:inputError", "First input must be a struct.");
    }

    int Mode = (int)mxGetScalar(mxGetField(prhs[0], 0, "Mode"));

    if (Mode == 0) {
        // ===================== Memory Initialization & Parameter Update =====================
        mexPrintf("Entering memory initialization and parameter update process\n");
        
        // Extract simulation parameters
        double Fs = mxGetScalar(mxGetField(prhs[0], 0, "Fs"));
        int Channels = (int)mxGetScalar(mxGetField(prhs[0], 0, "Channels"));
        double ArrayPitch = mxGetScalar(mxGetField(prhs[0], 0, "ArrayPitch"));
        double C0 = mxGetScalar(mxGetField(prhs[0], 0, "C0"));
        double T0 = mxGetScalar(mxGetField(prhs[0], 0, "T0"));
        int maxLen = (int)mxGetScalar(mxGetField(prhs[0], 0, "maxLen"));
        double F0 = mxGetScalar(mxGetField(prhs[0], 0, "F0"));
        mxArray* BandFiltersCoe = mxGetField(prhs[0], 0, "BandFiltersCoe");
        mxArray* DemodFiltersCoe = mxGetField(prhs[0], 0, "DemodFiltersCoe");

        // Update simulation parameter structure
        mSimPara.Mode = Mode;
        mSimPara.Fs = Fs;
        mSimPara.Channels = Channels;
        mSimPara.ArrayPitch = ArrayPitch;
        mSimPara.C0 = C0;
        mSimPara.T0 = T0;
        mSimPara.maxLen = maxLen;
        mSimPara.F0 = F0;

#if ENABLE_DEBUG_PRINT
        mexPrintf("mSimPara parameters: Mode=%d  Fs=%f  Channels=%d  ArrayPitch=%f   C0=%f  T0=%f  maxLen=%d  F0=%f \n", 
                  Mode, Fs, Channels, ArrayPitch, C0, T0, maxLen, F0);
#endif   
        
        // Check and copy BandFiltersCoe array
        if (BandFiltersCoe) {
            mwSize numElements = mxGetNumberOfElements(BandFiltersCoe);
            mSimPara.BandFiltersCoeSize = (int)numElements;
            float* bandCoe = (float*)mxGetPr(BandFiltersCoe);
#if ENABLE_DEBUG_PRINT
            mexPrintf("BandFiltersCoe has %zu elements\n", numElements);
#endif
            for (mwSize i = 0; i < numElements && i < 500; ++i) {
                mSimPara.BandFiltersCoe[i] = bandCoe[i];
            }
        }
        
        // Check and copy DemodFiltersCoe array
        if (DemodFiltersCoe) {
            mwSize numElements = mxGetNumberOfElements(DemodFiltersCoe);
            mSimPara.DemodFiltersCoeSize = (int)numElements;
            float* demodCoe = (float*)mxGetPr(DemodFiltersCoe);
#if ENABLE_DEBUG_PRINT
            mexPrintf("DemodFiltersCoe has %zu elements\n", numElements);
#endif
            for (mwSize i = 0; i < numElements && i < 500; ++i) {
                mSimPara.DemodFiltersCoe[i] = demodCoe[i];
            }
        }

        // Extract beamforming parameters
        if (nrhs < 2) {
            mexErrMsgIdAndTxt("PlaneBeam:inputError", "Second input (beamforming parameters) required for Mode 0.");
        }
        
        mxArray* PlaneAngletmp = mxGetField(prhs[1], 0, "PlaneAngle");
        double FrameNum = mxGetScalar(mxGetField(prhs[1], 0, "FrameNum"));
        double Fnum = mxGetScalar(mxGetField(prhs[1], 0, "Fnum"));
        mxArray* Pvectortmp = mxGetField(prhs[1], 0, "Pvector");
        mxArray* Xvectortmp = mxGetField(prhs[1], 0, "Xvector");
        mxArray* Zvectortmp = mxGetField(prhs[1], 0, "Zvector");

        mBeamPara.FrameNum = (int)FrameNum;
        mBeamPara.Fnum = (int)Fnum;
        
#if ENABLE_DEBUG_PRINT
        mexPrintf("mBeamPara parameters  FrameNum = %f  Fnum = %f \n", FrameNum, Fnum);
#endif
        
        // Check and copy PlaneAngle array
        if (PlaneAngletmp) {
            mwSize numElements = mxGetNumberOfElements(PlaneAngletmp);
            mBeamPara.PlaneAngleSize = (int)numElements;
            float* data = (float*)mxGetPr(PlaneAngletmp);
#if ENABLE_DEBUG_PRINT
            mexPrintf("PlaneAngle has %zu Angle\n", numElements);
#endif
            for (mwSize i = 0; i < numElements && i < 64; ++i) {
#if ENABLE_DEBUG_PRINT
                mexPrintf("PlaneAngle[%zu] = %f\n", i, data[i]);
#endif
                mBeamPara.PlaneAngle[i] = data[i];
            }
        }
        
        // Check and copy Pvector array
        if (Pvectortmp) {
            mwSize numElements = mxGetNumberOfElements(Pvectortmp);
            mBeamPara.PvectorSize = (int)numElements;
            float* data = (float*)mxGetPr(Pvectortmp);
#if ENABLE_DEBUG_PRINT
            mexPrintf("Pvectort has %zu elements\n", numElements);
#endif
            for (mwSize i = 0; i < numElements && i < 1000; ++i) {
                mBeamPara.Pvector[i] = data[i];
            }
        }
        
        // Check and copy Xvector array
        if (Xvectortmp) {
            mwSize numElements = mxGetNumberOfElements(Xvectortmp);
            mBeamPara.XvectorSize = (int)numElements;
            float* data = (float*)mxGetPr(Xvectortmp);
#if ENABLE_DEBUG_PRINT
            mexPrintf("Xvectort has %zu elements\n", numElements);
#endif
            for (mwSize i = 0; i < numElements && i < 2000; ++i) {
                mBeamPara.Xvector[i] = data[i];
            }
        }
        
        // Check and copy Zvector array
        if (Zvectortmp) {
            mwSize numElements = mxGetNumberOfElements(Zvectortmp);
            mBeamPara.ZvectorSize = (int)numElements;
            float* data = (float*)mxGetPr(Zvectortmp);
#if ENABLE_DEBUG_PRINT
            mexPrintf("Zvectort has %zu elements\n", numElements);
#endif
            for (mwSize i = 0; i < numElements && i < 2000; ++i) {
                mBeamPara.Zvector[i] = data[i];
            }
        }
        
        // Initialize CUDA device and memory
        int result = g_PlaneBeam_Initialize(&mSimPara, &mBeamPara);
        if (result != 0) {
            HandleError(result);
        }
        
    } else if (Mode == 1) {
        // ===================== Beamforming Computation =====================
        if (nrhs < 4) {
            mexErrMsgIdAndTxt("PlaneBeam:inputError", "Four input arguments required for Mode 1.");
        }
        
        // Validate input parameters
        if (!prhs[2] || !prhs[3]) {
            mexErrMsgIdAndTxt("PlaneBeam:inputError", "Invalid input data pointers.");
        }
        
        short* srcData = (short*)mxGetPr(prhs[2]);
        mxComplexSingle* result_bf_data = mxGetComplexSingles(prhs[3]);
        
        if (!srcData || !result_bf_data) {
            mexErrMsgIdAndTxt("PlaneBeam:inputError", "Failed to get data pointers from MATLAB arrays.");
        }
        
        // Calculate data size
        int dataSize = mSimPara.maxLen * mSimPara.Channels * mBeamPara.PlaneAngleSize * mBeamPara.FrameNum;
        int outputSize = mBeamPara.FrameNum * mBeamPara.XvectorSize * mBeamPara.ZvectorSize;
        
        if (dataSize <= 0 || outputSize <= 0) {
            mexErrMsgIdAndTxt("PlaneBeam:inputError", "Invalid data size: dataSize=%d, outputSize=%d", dataSize, outputSize);
        }
        
        // Allocate output data buffer
        ComplexFloat* outputData = nullptr;
        try {
            outputData = new ComplexFloat[outputSize];
            if (!outputData) {
                mexErrMsgIdAndTxt("PlaneBeam:memoryError", "Failed to allocate output buffer.");
            }
        } catch (const std::bad_alloc& e) {
            mexErrMsgIdAndTxt("PlaneBeam:memoryError", "Memory allocation failed: %s", e.what());
        }
        
        // Execute beamforming computation
        int result = 0;
        try {
            result = g_PlaneBeam_Process(srcData, outputData, dataSize);
        } catch (...) {
            delete[] outputData;
            mexErrMsgIdAndTxt("PlaneBeam:runtimeError", "Exception during beamforming computation.");
        }
        
        if (result != 0) {
            delete[] outputData;
            HandleError(result);
        }
        
        // Convert results to MATLAB format
        try {
            for (int i = 0; i < outputSize; i++) {
                result_bf_data[i].real = outputData[i].real;
                result_bf_data[i].imag = outputData[i].imag;
            }
        } catch (...) {
            delete[] outputData;
            mexErrMsgIdAndTxt("PlaneBeam:runtimeError", "Exception during data conversion.");
        }
        
        delete[] outputData;
        
    } else if (Mode == 9) {
        // ===================== Memory Release =====================
        mexPrintf("Entering memory release process\n");
        UnloadDLL();
        
    } else {
        // ===================== Invalid Mode =====================
        mexPrintf("Current mode setting is invalid\n");
    }
}

// MEX cleanup function
void mexAtExit(void) {
    UnloadDLL();
}
