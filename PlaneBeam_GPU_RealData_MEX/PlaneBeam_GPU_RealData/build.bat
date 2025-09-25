@echo off
echo Building PlaneBeam GPU RealData with DLL Architecture...

REM 设置环境变量
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set MATLAB_ROOT=D:\software\MATLAB

REM 检查CUDA路径
if not exist "%CUDA_PATH%" (
    echo Error: CUDA path not found at %CUDA_PATH%
    echo Please update the CUDA_PATH variable in this script
    pause
    exit /b 1
)

REM 检查MATLAB路径
if not exist "%MATLAB_ROOT%" (
    echo Error: MATLAB path not found at %MATLAB_ROOT%
    echo Please update the MATLAB_ROOT variable in this script
    pause
    exit /b 1
)

echo Building CUDA DLL...
msbuild PlaneBeam_GPU_RealData\PlaneBeam_CUDA_DLL\PlaneBeam_CUDA_DLL.vcxproj /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to build CUDA DLL
    pause
    exit /b 1
)

echo Building MEX wrapper...
msbuild PlaneBeam_GPU_RealData\PlaneBeam_GPU_RealData.vcxproj /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to build MEX wrapper
    pause
    exit /b 1
)

echo Build completed successfully!
echo.
echo Output files:
echo   - bin\x64\Release\PlaneBeam_CUDA_DLL.dll
echo   - bin\x64\Release\PlaneBeam_GPU_RealData.mexw64
echo.
echo To test the DLL, run: test_dll.exe
echo To use in MATLAB, ensure the DLL is in your PATH or current directory
pause
