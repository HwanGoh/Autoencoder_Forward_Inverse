% QPAT_EWEDGM2D_Driver.m is the main script file for the quantitative photoacoustic
% tomography inverse problem using the elastic wave equation and discontinuous Galerkin method.
%
% % Hwan Goh, 17/09/2017, University of Auckland, New Zealand

%=== Some useful command lines for running this code over the server ===%
% aitken.math.auckland.ac.nz
% ssh maclaurin.math
% addpath('QPAT_EWE_DGM2D_Codes_Hwan')
% run QPAT_EWEDGM2D_Driver
% cd /gpfs1m/projects/nesi00452/QPAT_EWE_DGM2D_Codes_Hwan

close all
clear all
clc
restoredefaultpath
format long
warning('off','all')
addpath(genpath('../Generate_Wave_Data'))

%adding paths one by one because the servers don't like the genpath command 
addpath('QPAT_EWE_DGM2D_Codes_Hwan')
addpath('QPAT_EWE_DGM2D_Codes_Hwan/AcousticForwardProblemDGM')
addpath('QPAT_EWE_DGM2D_Codes_Hwan/HesthavenAndWarburtonCodes')
addpath('QPAT_EWE_DGM2D_Codes_Hwan/MATFiles')
addpath('QPAT_EWE_DGM2D_Codes_Hwan/MeshGenerationAndInterpolation')
addpath('QPAT_EWE_DGM2D_Codes_Hwan/Miscellaneous')

%% =======================================================================%
%                        Demonstration Properties
%=========================================================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate Parameters and Mesh or Load Existing? %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=== Mesh ===%
RunOptions.GenerateMesh = 0; %Whether or not to generate mesh

%% %%%%%%%%%%%%%%%%%%%%
%%% Mesh Properties %%%
%%%%%%%%%%%%%%%%%%%%%%%
%=== Which Mesh ===%
RunOptions.UseTrelisMesh = 1; %Use Trelis generated mesh, set these properties in Trelis_Skull2D_JournalGenerator.m

%=== Sensor and Lights Source Properties ===%
RunOptions.NumberofSensorsOnOneBoundaryEdge = 10; %Number of sensors on one boundary edge of the domain
RunOptions.UseFullDomainData = 0; %Output of forward problem sensory data consisting of the full domain; still set RunOptions.FullqVectorData = 1
RunOptions.FullqVectorData = 0; %Output of forward problem sensory data of full q vector
RunOptions.VelocitiesData = 1; %Output of forward problem sensory data of velocities, only works when sensors are placed on the boundary

%=== Trelis Mesh Properties ===%
RunOptions.TrelisMeshDElementSize = '0009'; %Entry for generating Trelis data mesh, main purpose is for the file name when saving
RunOptions.BoundaryCondition = 'Neumann';

%% %%%%%%%%%%
%%% Prior %%%
%%%%%%%%%%%%%
RunOptions.N_Samples = 50000;
%=== Initial Condition ===%
Prior.Exp_h = 0.4; %Expected Value of h, Default = 70
Prior.AC_Var_h = 0.1^2; %Variance of p_0
Prior.AC_Corr_h = 0.0015; %Correlation Length of h, Default: 0.0001;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Acoustic Forward Problem %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=== Domain Type ===%
RunOptions.FluidDomainWithSolidLayerMeshD = 0; %Use Fluid domain with solid layer representing the skull for data mesh
RunOptions.FluidMeshD = 1; %Use purely fluid domain for data mesh
RunOptions.SolidMeshD = 0; %Use purely solid domain for data mesh

%=== Acoustic Parameters ===% (Mitsuhasi: In fluid: cp = 1500 m/s, cs = 0. In skull, cp = 3000 m/s and cs = 1500 m/s.)
RunOptions.TestAcousticParameters = 1; %Use test acoustic parameters
RunOptions.RealAcousticParameters = 0; %Use realistic acoustic parameters
if RunOptions.TestAcousticParameters == 1;
RunOptions.AcousticMediumWaveSpeed_cp = sqrt(2);
RunOptions.ElasticMediumWaveSpeed_cp = sqrt(2);
RunOptions.ElasticMediumWaveSpeed_cs = sqrt(1/2);
RunOptions.AcousticMediumDensity = 1;
RunOptions.ElasticMediumDensity = 1;
RunOptions.ElasticLamemu = (RunOptions.ElasticMediumWaveSpeed_cs^2)*RunOptions.ElasticMediumDensity; %also known as the shear modulus, cs = sqrt(mu/rho) => mu = cs^2*rho
RunOptions.ElasticLamelambda = RunOptions.ElasticMediumWaveSpeed_cp^2*RunOptions.ElasticMediumDensity - 2*RunOptions.ElasticLamemu; %cp = sqrt{(lambda + 2mu)/rho)} => lambda = cp^2*rho - 2*mu
end
if RunOptions.RealAcousticParameters == 1;
RunOptions.AcousticMediumWaveSpeed_cp = 1500; %default is 1500 m/s: "Mitsuhashi: A forward-adjoint pair based on the elastic wave equation for use in transcranial photoacoustic computed tomography"
RunOptions.ElasticMediumWaveSpeed_cp = 3000; %default is 3000 m/s: "Mitsuhashi: A forward-adjoint pair based on the elastic wave equation for use in transcranial photoacoustic computed tomography"
RunOptions.ElasticMediumWaveSpeed_cs = 1500; %default is 1500 m/s: "Mitsuhashi: A forward-adjoint pair based on the elastic wave equation for use in transcranial photoacoustic computed tomography"
RunOptions.AcousticMediumDensity = 1000; %default is 1000 kg/m^3: "Mitsuhashi: A forward-adjoint pair based on the elastic wave equation for use in transcranial photoacoustic computed tomography"
RunOptions.ElasticMediumDensity = 1850; %default is 1850 kg/m^3: "Mitsuhashi: A forward-adjoint pair based on the elastic wave equation for use in transcranial photoacoustic computed tomography"
RunOptions.ElasticLamemu = (RunOptions.ElasticMediumWaveSpeed_cs^2)*RunOptions.ElasticMediumDensity; %also known as the shear modulus, cs = sqrt(mu/rho) => mu = cs^2*rho
RunOptions.ElasticLamelambda = RunOptions.ElasticMediumWaveSpeed_cp^2*RunOptions.ElasticMediumDensity - 2*RunOptions.ElasticLamemu; %cp = sqrt{(lambda + 2mu)/rho)} => lambda = cp^2*rho - 2*mu
end

%=== DGM Properties ===%
%=== Initial Conditions ===%
RunOptions.p0Test1 = 0; %Test case 1
RunOptions.p0Test2 = 1; %Test case 2
RunOptions.GaussianWidth = 1e5;%(1e11); %Increase to decrease width of the Gaussian
%=== RHS Computation ===%
RunOptions.DGMPolyOrder = 2; %Polynomial order used for approximation

%=== Time-Stepping Properties ===%
RunOptions.TimeStepUseCFLTimo = 1/2; %Set CFL condition for Timo's time step size, set to 0 if you don't want to use
RunOptions.TimeStepSizeLSERK4 = 1e-6; %Set own value for time step size for LSERK4 time stepping, set to 0 if you don't want to use
if RunOptions.TestAcousticParameters == 1;
    RunOptions.FinalTime = 0.008; %Final Time
end
if RunOptions.RealAcousticParameters == 1;
    RunOptions.FinalTime = 0.000004; %Final Time
end

%% %%%%%%%%%%%%%%%%%
%%% Adding noise %%%
%%%%%%%%%%%%%%%%%%%%
RunOptions.AddNoise = 0; %Add noise?
RunOptions.NoiseLevel = 0.001; %Scalar multiplier of noise draws
RunOptions.NoiseMinMax = 0; %Use max minus min of data
RunOptions.NoiseMinMaxS = 0; %Use max minus min of data at each sensor
RunOptions.NoiseMax = 1; %Use max of all data
RunOptions.NoiseMaxS = 0; %Use max of data at each sensor

%% %%%%%%%%%%%%%
%%% Plotting %%%
%%%%%%%%%%%%%%%%
PLOT.MeshD=1; %Data Mesh
PLOT.PriorSamples=1; %Plot draws from prior
PLOT.WaveDGMForwardMesh=1; %Plot Mesh for Acoustic Forward Problem
PLOT.DGMForward=1; %Plot DGM Generated Forward Acoustic Data
PLOT.DGMForwardQuiver=0; %Plot DGM Generated Forward Acoustic Data using Quiver
PLOT.DGMForwardSensorData=0; %Plot DGM Generated Forward Acoustic Data
PLOT.Noise=0; %Plot noisy data
PLOT.DGMPlotBirdsEyeView = 1; %Use birds eye view for DGM plots
PLOT.InitialPressurezAxis = [0 1]; %z axis limits for plotting QPAT elastic wave propogation
PLOT.DGMPlotUsezLim = 0; %Use z axis limits for DGM plots
PLOT.DGMPlotzAxis = [0 1]; %z axis limits for plotting QPAT elastic wave propogation
PLOT.HoldColourAxis = 1; %Hold colour axis
PLOT.ColourAxis = [0 1]; %Colour axis

%=== Pre-define Figures ===%
DefineFigures

%% =======================================================================%
%                 Generate and Plot Mesh and Parameters
%=========================================================================%
if RunOptions.GenerateMesh == 1
    %=== Using Trelis Generated Mesh ===%
    if RunOptions.UseTrelisMesh == 1
        MeshD = QPAT_TrelisMesh2DGenerator(RunOptions,'Trelis_QPATMesh_Data.inp',3,1);
    end
    %=== Generating the Rectangular Mesh Using My Code ===%
    if RunOptions.UseMyRectangularMesh == 1
        QPAT_RectangularMesh2DGenerator
    end
else
    %=== Using Trelis Generated Mesh ===%
    RunOptions.LoadFileNameMeshD = sprintf('Mesh-%s',RunOptions.TrelisMeshDElementSize);
    load(RunOptions.LoadFileNameMeshD);
    MeshD = Mesh; 
    clear Mesh
end
%keyboard %keyboard here to save newly generated mesh

%% =======================================================================%
%                         Display Selected Options
%=========================================================================%
if RunOptions.UseTrelisMesh == 1
    FilenamesofRunOptions  
    %=== Displayed Text ===%
    printf(['MeshD Number of Elements: ' num2str(MeshD.N_Elm)]);
    printf(['MeshD Number of Nodes: ' num2str(MeshD.N_Nodes)]);
    printf(['Domain Types: ' RunOptions.SaveFileNameDataDomain]);
    printf(['Noise Level: 0.' num2str(RunOptions.SaveFileNameNoiseLevel) '%']);
    printf(['Number of Sensors On One Boundary Edge: ' num2str(RunOptions.NumberofSensorsOnOneBoundaryEdge)]);
    printf(['Final Time: ' num2str(RunOptions.FinalTime)]);
    printf(['\nSave File Name: ' RunOptions.SaveFileName]);
end

%% =======================================================================%
%                          Testing Prior Samples
%=========================================================================%
%[Prior.L_pr,Prior.traceCov_h,Cov_pr,~] = SmoothnessPrior_AutoCorr(MeshD.Nodes,Prior.Exp_h,Prior.AC_Var_h,Prior.AC_Corr_h,2,PLOT);

%% =======================================================================%
%                           Forward Problem
%=========================================================================%
                          %==================%
                          %    Setting Up    %
                          %==================%                            
%%%%%%%%%%%%%%%%%%
%%% Set Up DGM %%%
%%%%%%%%%%%%%%%%%%
[DGMMeshD, PrecomputedIntrplteObjectsD] = EWE_DGM2D_Setup(RunOptions,MeshD,RunOptions.FluidMeshD,RunOptions.SolidMeshD,RunOptions.FluidDomainWithSolidLayerMeshD);
ConstructDGMSensors
PlotDGMMesh
AcousticForwardTimeStepsize
PLOT.TRI_DGMMeshD=delaunay(DGMMeshD.x,DGMMeshD.y); %To be used later when plotting wave propagation
DGMMeshD.pinfo = EWE_DGM2D_PrecomputeUpwindFluxPNonConf(RunOptions,DGMMeshD.pinfo,DGMMeshD.Norder,DGMMeshD.rho,DGMMeshD.lambda,DGMMeshD.mu,MeshD.DomainIndices,RunOptions.FluidDomainWithSolidLayerMeshD,RunOptions.SolidMeshD);             

                         %====================%
                         %    Computations    %
                         %====================%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Construct Single Test Case  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%              
%QPAT_EWE_DGM2D_AcousticForward

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Construct Samples  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%   
[hAS_FEM,vxSamplesDataTimeSteps,vySamplesDataTimeSteps] = EWE_DGM2D_ConstructSamples(RunOptions,MeshD.Nodes,MeshD.Elements,DGMMeshD.x,DGMMeshD.y,DGMMeshD.Np,DGMMeshD.K,PrecomputedIntrplteObjectsD,DGMMeshD.pinfo,DGMMeshD.rho,DataVrblsWave.SensorsD,dt,Prior,PLOT);

%=== Saving Samples ===%
save(RunOptions.SaveFileNameSamples,'hAS_FEM','vxSamplesDataTimeSteps','vySamplesDataTimeSteps','-v7.3')


