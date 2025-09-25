%% =========================================================================
% PlatformBeam_GPU_Demo.m
% =========================================================================
% Description: GPU-based platform beamforming demonstration program
% Function: Implementing ultrasound imaging beamforming algorithm with CUDA acceleration
% Author: CTDXXS
% Email: wuwentao@mail.ioa.ac.cn
% Date: 2025-09-15
% Version: V1.0.0
%
% Instructions:
% 1. Requires GPU device with CUDA support
% 2. Depends on PlaneBeam_GPU_RealData.mexw64 file (open-source implementation)
%    The PlaneBeam_GPU_RealData.mexw64 is an open-source CUDA-based implementation
%    for ultrasound beamforming, freely available for research and educational purposes.
% 3. Requires filter coefficient files: BS_50M_10_18_001.mat and LP_50M_2M_001.mat
% 4. Requires raw data file: 20250702_0_004.mat
%% =========================================================================

clc;clear all;close all;
%% --Import Filters
load BP_50M_10_18_001.mat;
BPF = single(Num);
load LP_50M_2M_001.mat;
LPF = single(Num);

%% --Configure Parameters
% Create SimPara structure
SimPara.Mode = 0;   %--0: Parameter update mode (including memory initialization), 1: Calculation mode, 9: Memory release mode

SimPara.Fs = 50e6;% Sampling frequency
SimPara.Channels = 64;% Number of channels
SimPara.Fs = 50e6;% Sampling frequency
SimPara.F0 = 15e6;% Signal center frequency
SimPara.ArrayPitch = 100e-6;% Element pitch, unit: m
SimPara.C0 = 1540;% Sound velocity
SimPara.Ts = 1/SimPara.Fs; % Sampling period
SimPara.T0 = 2e-6; % Data start time
SimPara.maxLen = 1024; % Maximum data length (points)
SimPara.F0 = 15e6; % Signal center frequency
SimPara.BandFiltersCoe = BPF;% Bandpass filter coefficients (note: even length and symmetric), maximum length 200
SimPara.DemodFiltersCoe = LPF;% Lowpass filter coefficients (note: even length and symmetric), maximum length 200


%--Generate parameters
N_elements = SimPara.Channels;
pitch = SimPara.ArrayPitch;
Num=1:N_elements;
Pvector0 = single((Num-N_elements/2)*pitch-1/2*pitch);

dx=0.05/1000;
dz=0.05/1000;

Xvector0 = single(-5/1000:dx:5/1000);% Horizontal axis for imaging, unit: m, size: 1xM
Zvector0 = single(1/1000:dz:10/1000);% Vertical axis for imaging, unit: m, size: 1xN

%-Transmission angle parameters
start_angle=-6.0/180*pi;
end_angle=6.0/180*pi;
deta_sita=12/(7-1)/180*pi;  %--Using the number of angles configured earlier

Tx_Anglepara = zeros(1,7,"single");
Tx_Anglepara=start_angle:deta_sita:end_angle;


% Beamforming parameters
BeamPara.PlaneAngle=single(Tx_Anglepara);%single(0/180*pi);% Plane wave incident angle
BeamPara.FrameNum = 200;% Number of frames
BeamPara.Fnum=1;% F-number for beamforming
BeamPara.Pvector=Pvector0;% Element coordinates, unit: m, size: 1xSimPara.Channels
BeamPara.Xvector=Xvector0;% Horizontal axis for imaging, unit: m, size: 1xM
BeamPara.Zvector=Zvector0;% Vertical axis for imaging, unit: m, size: 1xN

X_Size = length(Xvector0);
Z_Size = length(Zvector0);

% Load data
load 20250702_0_004.mat;

RF_data = srcDataBuffer;%[srcdata,srcdata,srcdata,srcdata];


figure;
for ii=1:N_elements
    plot(RF_data(:,ii)+ii*200);
    hold on;
end

real_part_rand = rand(Z_Size, X_Size,BeamPara.FrameNum,'single'); 
imag_part_rand = rand(Z_Size, X_Size,BeamPara.FrameNum,'single'); % Create real and imaginary parts of random arrays, with values uniformly distributed between 0 and 1
ImagIQ = complex(real_part_rand, imag_part_rand); % Combine real and imaginary parts into a complex array


%% Call Functions
% 1. Initialize
PlaneBeam_GPU_RealData(SimPara,BeamPara);

% 2. Compute
SimPara.Mode = 1;   %--0: Parameter update mode (including memory initialization), 1: Calculation mode, 9: Memory release mode
PlaneBeam_GPU_RealData(SimPara,BeamPara,RF_data,ImagIQ);

% 3. Release memory
SimPara.Mode = 9;   %--0: Parameter update mode (including memory initialization), 1: Calculation mode, 9: Memory release mode
PlaneBeam_GPU_RealData(SimPara,BeamPara,RF_data);

%% 3. Transform and Imaging        
N_elements=SimPara.Channels;
b=abs(ImagIQ(:,:,1));
b=b/max(b(:));
dynamic=60;
temp=(log10(b)*20+dynamic)/dynamic;
temp(temp<0)=0;
temp=temp*255;

x=BeamPara.Xvector;
z=BeamPara.Zvector;
start_x=x(1);
image_size_x=x(end)-x(1);
start_depth_z=z(1);
image_size_z=z(end)-z(1);

xLen=length(x);
zLen=length(z);
Nx=xLen;
Nz=zLen;
x_1=((1:Nx)/Nx*image_size_x+start_x)*1000;
z_1=((1:Nx)/Nx*image_size_z+start_depth_z)*1000;


figure;
I_lowpass = temp;
imagesc(x_1,z_1,I_lowpass); 
colormap gray;
xlabel('Horizontal Distance (mm)');
ylabel('Depth (mm)');

clear PlaneBeam_GPU_RealData;