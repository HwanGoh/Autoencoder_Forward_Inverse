function [hAS_FEM,vxSamplesDataTimeSteps,vySamplesDataTimeSteps]=EWE_DGM2D_ConstructSamples(RunOptions,MeshNodes,MeshElements,x,y,Np,K,PrecomputedIntrplteObjects,pinfo,DGMMeshrho,Sensors,dt,Prior,PLOT)

% EWE_DGM2D_ConstructSamples computes the data samples
%
% Inputs:
%   RunOptions:
%              NumberofTimeSteps
%              N_Samples - number of samples we wish to use to calculate approximation error
%   MeshNodes - Number of FEM Nodes by 2 array storing the coordinates of the nodes
%   MeshElements - Number of Elements by 3 array where the ith row contains the indices of the nodes in the ith element
%   x - x-coordinates of nodes on the DGM Mesh
%   y - y-coordinates of nodes on the DGM Mesh
%   Np - Number of nodes per element on the DGM Mesh
%   K - Number of elements
%   PrecomputedIntrplteObjects:
%                             Objects that depend on the inverse Mesh nodes. May have been computed earlier and so can 
%                             be called here to avoid repeating computations. Set to 0 in function call if you
%                             want to compute new objects for a new mesh.
%   pinfo - Invesion mesh information regarding p-refinement for p-nonconforming meshes
%   Sensors: 1 by number of sensors array containing 3 by 1 cells 
%             - id: Np by 1 array containing the indices of the nodes of the element the sensor is contained in
%             - xy: coordinates of the sensor
%             - l_iatsensor: 1 by Np array representing the local basis function; [l_1(r_0,s_0),l_2(r_0,s_0),...,l_Np(r_0,s_0)] where (r_0,s_0) is such that x^k(r_0,s_0) is the coordinates of a sensor
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
%      hAS_FEM - Prior model sample draws, mainly used for QR
%                construction of approximation error statistics
%      vxSamplesDataTimeSteps, vySamplesDataTimeSteps - velocirt samples
%
% Hwan Goh 15/01/2018, University of Auckland, New Zealand
%          27/07/2018 - huge overhaul of approximation error codes
%          12/11/2019, Oden Institute for Computational Sciences and Engineering, United States of America

PLOT.DGMForward = 1; %Suppress plotting of wave propagation
PLOT.PriorSamples = 1; %Suppress plotting of prior samples
PLOT.DGMForwardSensorData = 0; %Suppress plotting of sensory data
N_Samples = RunOptions.N_Samples;
NumberofSensors = size(Sensors,2);
NumberofTimeSteps = RunOptions.NumberofTimeSteps;
printf([num2str(N_Samples) ' samples to be computed for approximation error']);

%=========================================================================%
%                         Accurate Model Samples
%=========================================================================%
%=== Absorbed Energy Density Samples ===%
printf('Computing samples of accurate model, ACPrior Draws');
[~,~,~,hAS_FEM] = SmoothnessPrior_AutoCorr(MeshNodes,Prior.Exp_h,Prior.AC_Var_h,Prior.AC_Corr_h,N_Samples,PLOT);

%=== Forward Data Samples ===%
vxSamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);
vySamplesDataTimeSteps = zeros(NumberofSensors*NumberofTimeSteps,N_Samples);

%=== Accurate Forward Data ===%
for n=1:N_Samples
    printf(['\nComputing Accurate Model Sample ' num2str(n) ' of ' num2str(N_Samples)]);
    printf(['For the Case ' RunOptions.SaveFileNameSamples]);
    [hS_DGM, ~] = IntrplteOver2DTriangulatedMesh(size(MeshElements,1),MeshNodes,hAS_FEM(:,n),x,y,Np*K,PrecomputedIntrplteObjects);
    p0S_DGM = reshape(hS_DGM,Np,K)./(DGMMeshrho);
    [~,~,~,vxSTimeSteps,vySTimeSteps] = EWE_DGM2D_LSExpRK4(RunOptions,p0S_DGM,x,y,Np,K,pinfo,dt,PLOT);
    vxSDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    vySDataTimeSteps = sparse(NumberofSensors,NumberofTimeSteps);
    %=== Interpolation to Form Sensory Data ===%
    for t=1:NumberofTimeSteps
        for s=1:NumberofSensors
            vxSDataTimeSteps(s,t) = Sensors{s}.l_iatsensor*vxSTimeSteps(Sensors{s}.id,t);
            vySDataTimeSteps(s,t) = Sensors{s}.l_iatsensor*vySTimeSteps(Sensors{s}.id,t);
        end
    end
    vxSamplesDataTimeSteps(:,n) = vxSDataTimeSteps(:);
    vySamplesDataTimeSteps(:,n) = vySDataTimeSteps(:);
end
if RunOptions.FullqVectorData == 1;
clear T11ASTimeSteps T22ASTimeSteps T12ASTimeSteps T11ASDataTimeSteps T22ASDataTimeSteps T12ASDataTimeSteps
end
clear vxSTimeSteps vySTimeSteps vxSDataTimeSteps vySDataTimeSteps

