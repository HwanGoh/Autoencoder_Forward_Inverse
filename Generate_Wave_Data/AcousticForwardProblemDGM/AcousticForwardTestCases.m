%=== Test Case 1 ===%
if RunOptions.p0Test1 == 1;
    DataVrblsWave.p0DGM = zeros(DGMMeshD.K*DGMMeshD.Np,1);
    for ii=1:DGMMeshD.K*DGMMeshD.Np;
        if norm([DGMMeshD.x(ii),DGMMeshD.y(ii)] - [0.005,0.005],2)<0.003
            DataVrblsWave.p0DGM(ii) = 0.4;
        end
    end
end
%=== Test Case 2 ===%
if RunOptions.p0Test2 == 1;
    DataVrblsWave.p0DGM = zeros(DGMMeshD.K*DGMMeshD.Np,1);
    for ii=1:DGMMeshD.K*DGMMeshD.Np;
        if norm([DGMMeshD.x(ii),DGMMeshD.y(ii)] - [0.005,0.005],2)<0.003
            DataVrblsWave.p0DGM(ii) = 0.4;
        end
        if abs(DGMMeshD.x(ii) - 0.001)<0.005 && abs(DGMMeshD.y(ii) - 0.001) < 0.001;
            DataVrblsWave.p0DGM(ii) = 0.4;
        end
        if abs(DGMMeshD.y(ii) - 0.001)<0.005 && abs(DGMMeshD.x(ii) - 0.001) < 0.001;
            DataVrblsWave.p0DGM(ii) = 0.4;
        end
    end
end
DataVrblsWave.p0DGM = reshape(DataVrblsWave.p0DGM,DGMMeshD.Np,DGMMeshD.K);
