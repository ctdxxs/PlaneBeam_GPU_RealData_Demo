# PlaneBeam_GPU_RealData

## Overview
PlaneBeam_GPU_RealData is a MATLAB-CUDA integration project that provides GPU-accelerated processing for planar beamforming applications. This project consists of a CUDA-accelerated DLL library and a MATLAB MEX wrapper, enabling high-performance planar beam processing directly from MATLAB.

## Features
- GPU-accelerated planar beamforming using NVIDIA CUDA
- Seamless integration with MATLAB via MEX interface
- Dynamic loading of CUDA-accelerated DLL
- CUDA device management and resource optimization
- Processing of real-world data from MATLAB environment

## Project Structure
```
├── PlaneBeam_GPU_RealData.sln       # Visual Studio solution file
├── PlaneBeam_GPU_RealData/          # Main project directory
│   ├── PlaneBeam_CUDA_DLL/          # CUDA-accelerated library
│   ├── PlaneBeam_GPU_RealData.vcxproj # Visual Studio project file
│   ├── PlaneBeam_GPU_RealData_MEX.cpp # MATLAB MEX interface
│   ├── build/                       # Build output directory
│   ├── build.bat                    # Build script
│   ├── matlab.targets               # MATLAB build targets
│   └── matlab.xml                   # MATLAB build configuration
└── README.md                        # Project documentation
```

## Dependencies
- MATLAB (with MEX support)
- NVIDIA CUDA Toolkit (v12.6 or compatible)
- Visual Studio (with C++ and CUDA development tools)
- Windows operating system

## Build Instructions
1. Ensure that CUDA Toolkit and MATLAB are installed on your system
2. Update the environment variables in `build.bat` to match your system configuration:
   ```batch
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
   set MATLAB_ROOT=D:\software\MATLAB
   ```
3. Run the build script:
   ```batch
   build.bat
   ```
4. Successful build will generate:
   - `bin\x64\Release\PlaneBeam_CUDA_DLL.dll`
   - `bin\x64\Release\PlaneBeam_GPU_RealData.mexw64`

## Usage in MATLAB
1. Ensure the built DLL and MEX files are in your MATLAB path or current working directory
2. Call the MEX function directly from MATLAB with appropriate parameters

## Key Components

### MEX Interface
The MEX interface (`PlaneBeam_GPU_RealData_MEX.cpp`) provides the following functionality:
- Dynamic loading of the CUDA DLL
- Initialization of CUDA environment and planar beam parameters
- Data processing from MATLAB inputs
- CUDA resource management and cleanup
- Error handling and reporting

### CUDA Library
The CUDA-accelerated library provides high-performance implementations of planar beamforming algorithms optimized for GPU execution.

## Authors
- CTDXXS
- Email: wuwentao@mail.ioa.ac.cn

## Version
V1.0.1

## Creation Date
September 15, 2024

## License
© 2024 CTDXXS. All rights reserved.

## Notes
- Ensure that your system has compatible NVIDIA GPU hardware with CUDA support
- The DLL path must be accessible from MATLAB when using the MEX function
- For optimal performance, use the latest NVIDIA drivers and CUDA Toolkit
