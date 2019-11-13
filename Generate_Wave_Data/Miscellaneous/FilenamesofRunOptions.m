%=== Filenames of Outputs ===%
if RunOptions.FluidDomainWithSolidLayerMeshD == 1 %For saving data
    RunOptions.SaveFileNameDataDomain = 'SL';
end
if RunOptions.FluidMeshD == 1 %For saving data
    RunOptions.SaveFileNameDataDomain = 'F';
end

%=== Noisy Data Properties ===%
if RunOptions.AddNoise == 0;
    RunOptions.NoiseLevel = 0;
end
TempString = num2str(RunOptions.NoiseLevel);
RunOptions.SaveFileNameNoiseLevel = sscanf(TempString(3:end),'%s');
if RunOptions.NoiseMinMax == 1
    RunOptions.SaveFileNameNoiseType = 'MinMax';
end
if RunOptions.NoiseMinMaxS == 1
    RunOptions.SaveFileNameNoiseType = 'MinMaxS';
end
if RunOptions.NoiseMax == 1
    RunOptions.SaveFileNameNoiseType = 'Max';
end
if RunOptions.NoiseMaxS == 1
    RunOptions.SaveFileNameNoiseType = 'MaxS';
end

%=== Parameter Type and Final Time ===%
if RunOptions.TestAcousticParameters == 1;
    RunOptions.SaveFileNameParameterType = 'TestPrmtrs';
    TempString = num2str(RunOptions.FinalTime);
    RunOptions.SaveFileNameFinalTime = sscanf(TempString(3:end),'%s');
end
if RunOptions.RealAcousticParameters == 1;
    RunOptions.SaveFileNameParameterType = 'RealPrmtrs';
    TempString = num2str(RunOptions.FinalTime);
    RunOptions.SaveFileNameFinalTime = sscanf(TempString(1),'%s');
    clear TempString
end

%=== Save File Name ===%
RunOptions.SaveFileName = sprintf('%s_%s_%sD_%s%s_%dSensors_%sFinalTime',RunOptions.SaveFileNameParameterType,RunOptions.SaveFileNameDataDomain,RunOptions.TrelisMeshDElementSize,RunOptions.SaveFileNameNoiseLevel,RunOptions.SaveFileNameNoiseType,RunOptions.NumberofSensorsOnOneBoundaryEdge,RunOptions.SaveFileNameFinalTime);
RunOptions.SaveFileNameSamples = sprintf('Samples-%s-%s-%sD-%dSensors-%sFinalTime-%dSamples',RunOptions.SaveFileNameParameterType,RunOptions.SaveFileNameDataDomain,RunOptions.TrelisMeshDElementSize,RunOptions.NumberofSensorsOnOneBoundaryEdge,RunOptions.SaveFileNameFinalTime,RunOptions.N_Samples);


