# GPU-Accelerated Ultrasound Beamforming Platform

## Overview
This project demonstrates a GPU-accelerated ultrasound beamforming algorithm implementation using CUDA. The platform provides real-time processing capabilities for ultrasound imaging applications.

## Features
- **GPU Acceleration**: Utilizes CUDA for high-performance beamforming computation
- **Real-time Processing**: Supports real-time ultrasound data processing
- **Flexible Configuration**: Configurable parameters for different imaging scenarios
- **Open Source**: The core beamforming library (PlaneBeam_GPU_RealData.mexw64) is open-source
- **MATLAB Integration**: Seamless integration with MATLAB for research and development

## System Requirements
- **Operating System**: Windows 64-bit
- **MATLAB**: R2023b or later
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 3.5 or higher)
- **CUDA Toolkit**: Version 10.0 or later
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 2GB free disk space

## Installation Guide

### 1. Prerequisites
Ensure you have the following installed:
- MATLAB R2023b or later
- NVIDIA GPU drivers (latest version recommended)
- CUDA Toolkit 10.0+

### 2. Setup Steps
1. Download and extract the project files to your desired directory
2. Open MATLAB and navigate to the project folder
3. Verify CUDA support by running `gpuDevice` in MATLAB command window
4. Run the demo script to test the installation

## File Structure
