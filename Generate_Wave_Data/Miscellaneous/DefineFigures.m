if PLOT.WaveDGMForwardMesh==1;
    PLOT.Figure_WaveDGMForwardMesh = figure;
    PLOT.Figure_WaveDGMForwardMesh_Title = 'Forward Mesh';
    movegui(PLOT.Figure_WaveDGMForwardMesh,'northwest');
    drawnow
end

if PLOT.DGMForward==1
    PLOT.Figure_WavePropagation_Data = figure;
    PLOT.Figure_WavePropagation_Data_Title = 'Elastic Wave Propagation';
    movegui(PLOT.Figure_WavePropagation_Data,'southwest');
    drawnow
end

if PLOT.DGMForwardQuiver == 1;
    PLOT.Figure_WavePropagationQuiver_Data = figure;
    PLOT.Figure_WavePropagation_Data_Title = 'Elastic Wave Propagation';
    drawnow
end

if RunOptions.UseFullDomainData ~= 1 && PLOT.DGMForwardSensorData == 1;
    PLOT.Figure_WavePropagation_SensorData = figure;
    PLOT.Figure_WavePropagation_SensorData_Title = 'Sensor Data';
    drawnow
end