%% =======================================================================%
%                           Plotting FEM Mesh
%=========================================================================%
if RunOptions.GeneratePrmtrs == 1;
    if PLOT.MeshD == 1
        figure(PLOT.Figure_MeshD)
        triplot(MeshD.Elements,MeshD.Nodes(:,1),MeshD.Nodes(:,2))
        hold on
        %=== Boundary Elements ===%
        for ii = MeshD.Bnd_ElmInd' %Change this to highlight specific elements
            coord=MeshD.Nodes(MeshD.Elements(ii,:),:);
            patch(coord(:,1),coord(:,2),[1 0 0])
        end
        %=== Numbering Nodes ===%
        %     for ii=1:MeshD.N_Nodes
        %         text(MeshD.Nodes(ii,1),MeshD.Nodes(ii,2),num2str(ii),'Color','r')
        %     end
        %=== Boundary Nodes ===%
        %     for ii = 1:size(MeshD.Bnd_NodeInd,1)
        %         text(MeshD.Nodes(MeshD.Bnd_NodeInd(ii),1),MeshD.Nodes(MeshD.Bnd_NodeInd(ii),2),num2str(MeshD.Bnd_NodeInd(ii)),'Color','r')
        %     end
        %=== Domains ===%
        %     for ii=1:MeshD.N_Elm
        %         clc
        %         if MeshD.DomainIndices(ii) == 3
        %             coord=MeshD.Nodes(MeshD.Elements(ii,:),:);
        %             patch(coord(:,1),coord(:,2),[1 0 0])
        %         end
        %     end
        axis 'image'
        title(PLOT.Figure_MeshD_Title,'FontWeight','bold')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plotting Pre-Generated Data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plotting targets and meshes when using pre-generated parameters
if RunOptions.GeneratePrmtrs == 0;
    %Data Mesh
    if PLOT.MeshD == 1;
        figure(PLOT.Figure_MeshD)
        triplot(MeshD.Elements,MeshD.Nodes(:,1),MeshD.Nodes(:,2))
        hold on
        %=== Boundary Elements ===%
        for ii = MeshD.Bnd_ElmInd' %Change this to highlight specific elements
            coord=MeshD.Nodes(MeshD.Elements(ii,:),:);
            patch(coord(:,1),coord(:,2),[1 0 0])
        end
%          %=== Solid Layer Elements ===%
%         for ii = 1:MeshD.N_Elm %Change this to highlight specific elements
%             if MeshD.DomainIndices(ii) == 2
%                 coord=MeshD.Nodes(MeshD.Elements(ii,:),:);
%                 patch(coord(:,1),coord(:,2),[1 0 1])
%             else
%                 coord=MeshD.Nodes(MeshD.Elements(ii,:),:);
%                 patch(coord(:,1),coord(:,2),[0 0 1])
%             end
%         end  
        %=== Numbering Nodes ===%
%         for ii=1:MeshD.N_Nodes
%             text(MeshD.Nodes(ii,1),MeshD.Nodes(ii,2),num2str(ii),'Color','r')
%         end
        %=== Domains ===%
%         for ii=1:MeshD.N_Elm %Change this to highlight specific elements
%             if MeshD.DomainIndices(ii) == 1
%                 coord=MeshD.Nodes(MeshD.Elements(ii,:),:);
%                 patch(coord(:,1),coord(:,2),[1 0 0])
%             end
%         end
        axis 'image'
        title(PLOT.Figure_MeshD_Title,'FontWeight','bold')
    end
end
clear coord