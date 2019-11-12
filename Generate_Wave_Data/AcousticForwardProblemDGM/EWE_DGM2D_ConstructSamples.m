function [hAS_FEM,hIS_FEM,errorT11,errorT22,errorT12,errorvx,errorvy]=EWE_DGM2D_ConstructSamples(RunOptions,MeshANodes,MeshAElements,MeshADimensns,MeshADomainIndices,xA,yA,NpA,KA,PrecomputedIntrplteObjectsA,pinfoA,DGMMeshANorder,DGMMeshAlambda,DGMMeshArho,SensorsA,dt,Prior,PLOT)

% EWE_DGM2D_ConstructSamples computes the samples required for calculating 
% the approximation error mean and approximation error covariance
%
% Inputs:
%   RunOptions:
%              NumberofTimeSteps
%              AEN_Samples - number of samples we wish to use to calculate approximation error
%   MeshANodes, MeshINodes - Number of FEM Nodes by 2 array storing the coordinates of the nodes
%   MeshAElements, MeshIElements - Number of Elements by 3 array where the ith row contains the indices of the nodes in the ith element
%   MeshADimensns - Width and height of the accurate mesh
%   xA, xI - x-coordinates of nodes on the DGM Mesh
%   yA, yI - y-coordinates of nodes on the DGM Mesh
%   NpA, NpI - Number of nodes per element on the DGM Mesh
%   KA, KI - Number of elements
%   PrecomputedIntrplteObjectsA,PrecomputedIntrplteObjectsI:
%                             Objects that depend on the inverse Mesh nodes. May have been computed earlier and so can 
%                             be called here to avoid repeating computations. Set to 0 in function call if you
%                             want to compute new objects for a new mesh.
%   pinfoA,pinfoI - Invesion mesh information regarding p-refinement for p-nonconforming meshes
%   SensorsA, SensorsI: 1 by number of sensors array containing 3 by 1 cells 
%                       - id: Np by 1 array containing the indices of the nodes of the element the sensor is contained in
%                       - xy: coordinates of the sensor
%                       - l_iatsensor: 1 by Np array representing the local basis function; [l_1(r_0,s_0),l_2(r_0,s_0),...,l_Np(r_0,s_0)] where (r_0,s_0) is such that x^k(r_0,s_0) is the coordinates of a sensor
%   dt - size of time steps
%   Prior:
%      corr_h - 2*1 vector = [correlation_x, correlation_y]. The larger the number,
%          the more points near to a marginalisation point are correlated
%          in the x and y direction.
%      bounds_h - approximately [h_min, h_max]
%      Exp_h - initial guess of h
%   PLOT - To plot or not to plot, that is the question
%
% Outputs:
%      hAS_FEM, hIS_FEM - Prior model sample draws, mainly used for QR
%                           construction of approximation error statistics
%      errorTij, errorvi - Sample errors
%
% Hwan Goh 15/01/2018, University of Auckland, New Zealand
%          27/07/2018 - huge overhaul of approximation error codes

PLOT.DGMForward = 0; %Suppress plotting of wave propagation
PLOT.PriorMargPoints = 0; %Suppress plotting of marginalisation points
PLOT.PriorSamples = 0; %Suppress plotting of prior samples
PLOT.DGMForwardSensorData = 0; %Suppress plotting of sensory data
N_Samples = RunOptions.AEN_Samples;
NumberofSensors = size(SensorsI,2);
NumberofTimeSteps = RunOptions.NumberofTimeSteps;
printf([num2str(N_Samples) ' samples to be computed for approximation error']);

if RunOptions.GenerateAndSave_AESamples == 1
%=========================================================================%
%                         Accurate Model Samples
%=========================================================================%
printf('Computing samples of accurate model');

%=== Absorbed Energy Density Samples ===%
printf('Computing samples of accurate model, ISPrior Draws');
[~,~,~,~,hAS_FEM]=SmoothnessPrior_Informative(MeshANodes,MeshAElements,2*MeshADimensns,Prior.InformSmooth_Bounds_h,Prior.InformSmooth_Corr_h,Prior.Exp_h,Prior.InformSmooth_Normalize,N_Samples,PLOT);

%=== Forward Data Samples ===%
if RunOptions.FullqVectorData == 1;
    T11ASamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);
    T22ASamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);
    T12ASamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);
else
    T11ASamplesDataTimeSteps = sparse(NumberofSensors*NumberofTimeSteps,N_Samples);
    T22ASamplesDataTimeSteps = sparse(NumberofSensors*NumberofTimeSteps,N_Samples);
    T12ASamplesDataTimeSteps = sparse(NumberofSensors*NumberofTimeSteps,N_Samples);
end
vxASamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);
vyASamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);

PLOT.TRI_DGMMeshD = PLOT.TRI_DGMMeshA; %This is for plotting forward elastic wave propagation on inversion mesh. FwrdFunction is called which is EWE_DGM2D_LSExpRK4. However, that function calls PLOT.TRI_DGMMeshD when plotting, so here we replace it with DGMMeshA but re-use the same name

%=== Accurate Forward Data ===%
for n=1:N_Samples
    printf(['\nComputing Accurate Model Sample ' num2str(n) ' of ' num2str(N_Samples)]);
    printf(['For the Case ' RunOptions.SaveFileNameAESamples]);
    [hAS_DGM, ~] = IntrplteOver2DTriangulatedMesh(size(MeshAElements,1),MeshANodes*(1/RunOptions.ScalingOptical),hAS_FEM(:,n),xA,yA,NpA*KA,PrecomputedIntrplteObjectsA);
    p0AS_DGM = RunOptions.LinearHeatExpansionCoeff*reshape(hAS_DGM,NpA,KA)./(DGMMeshArho*RunOptions.SpecificHeatCoeff);
    if RunOptions.TimeLSERK4 == 1;
        [T11ASTimeSteps,T22ASTimeSteps,T12ASTimeSteps,vxASTimeSteps,vyASTimeSteps] = EWE_DGM2D_LSExpRK4(RunOptions,p0AS_DGM,xA,yA,NpA,KA,pinfoA,dt,PLOT);
    end
    T11ASDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    T22ASDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    T12ASDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    vxASDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    vyASDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    %=== Interpolation to Form Sensory Data ===%
    for t=1:NumberofTimeSteps
        for s=1:NumberofSensors
            if RunOptions.FullqVectorData == 1;
                T11ASDataTimeSteps(s,t) = SensorsA{s}.l_iatsensor*T11ASTimeSteps(SensorsA{s}.id,t);
                T22ASDataTimeSteps(s,t) = SensorsA{s}.l_iatsensor*T22ASTimeSteps(SensorsA{s}.id,t);
                T12ASDataTimeSteps(s,t) = SensorsA{s}.l_iatsensor*T12ASTimeSteps(SensorsA{s}.id,t);
            end
            vxASDataTimeSteps(s,t) = SensorsA{s}.l_iatsensor*vxASTimeSteps(SensorsA{s}.id,t);
            vyASDataTimeSteps(s,t) = SensorsA{s}.l_iatsensor*vyASTimeSteps(SensorsA{s}.id,t);
        end
    end
    if RunOptions.FullqVectorData == 1;
        T11ASamplesDataTimeSteps(:,n) = T11ASDataTimeSteps(:);
        T22ASamplesDataTimeSteps(:,n) = T22ASDataTimeSteps(:);
        T12ASamplesDataTimeSteps(:,n) = T12ASDataTimeSteps(:);
    end
    vxASamplesDataTimeSteps(:,n) = vxASDataTimeSteps(:);
    vyASamplesDataTimeSteps(:,n) = vyASDataTimeSteps(:);
end
if RunOptions.FullqVectorData == 1;
clear T11ASTimeSteps T22ASTimeSteps T12ASTimeSteps T11ASDataTimeSteps T22ASDataTimeSteps T12ASDataTimeSteps
end
clear vxASTimeSteps vyASTimeSteps vxASDataTimeSteps vyASDataTimeSteps

%=== Saving and Loading Accurate Model Samples ===%
save(RunOptions.SaveFileNameAESamples,'hAS_FEM','T11ASamplesDataTimeSteps','T22ASamplesDataTimeSteps','T12ASamplesDataTimeSteps','vxASamplesDataTimeSteps','vyASamplesDataTimeSteps','-v7.3')
