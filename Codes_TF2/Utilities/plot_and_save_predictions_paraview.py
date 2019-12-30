import sys
sys.path.append('../..')

from paraview.simple import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Plot and Save Parameter and State                     #
###############################################################################
def plot_and_save_predictions_paraview(file_paths):
# =============================================================================
#     #=== Parameter Test ===#
#     pvd_load_filepath = file_paths.figures_savefile_name_parameter_test + '.pvd'
#     figure_save_filepath = file_paths.figures_savefile_name_parameter_test + '.png'
#     save_png(pvd_load_filepath, figure_save_filepath)
# =============================================================================
    
    #=== State Test ===#
    pvd_load_filepath = file_paths.figures_savefile_name_state_test + '.pvd'
    figure_save_filepath = file_paths.figures_savefile_name_state_test + '.png'
    save_png(pvd_load_filepath, figure_save_filepath)
    
# =============================================================================
#     #=== Parameter Predictions ===#
#     pvd_load_filepath = file_paths.figures_savefile_name_parameter_pred + '.pvd'
#     figure_save_filepath = file_paths.figures_savefile_name_parameter_pred + '.png'
#     save_png(pvd_load_filepath, figure_save_filepath)
#     
#     #=== State Predictions ===#
#     pvd_load_filepath = file_paths.figures_savefile_name_state_pred + '.pvd'
#     figure_save_filepath = file_paths.figures_savefile_name_state_pred+ '.png'
#     save_png(pvd_load_filepath, figure_save_filepath)
# 
# =============================================================================
###############################################################################
#                             Save Paraview Plot                              #
###############################################################################
def save_png(pvd_load_filepath, figure_save_filepath):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    
    # create a new 'PVD Reader'
    state_test_3D_varypvd = PVDReader(FileName='/home/hwan/Documents/Github_Codes/Autoencoder_Forward_Inverse/Figures/rev_maware_thermalfinvary_full_3D_hl5_tl3_hn500_relu_p10_d50000_b1000_e1000/parameter_test_3D_vary.pvd')
    
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [731, 546]
    
    # get color transfer function/color map for 'f_110'
    f_110LUT = GetColorTransferFunction('f_110')
    
    # get opacity transfer function/opacity map for 'f_110'
    f_110PWF = GetOpacityTransferFunction('f_110')
    
    # show data in view
    state_test_3D_varypvdDisplay = Show(state_test_3D_varypvd, renderView1)
    # trace defaults for the display properties.
    state_test_3D_varypvdDisplay.Representation = 'Surface'
    state_test_3D_varypvdDisplay.AmbientColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.ColorArrayName = ['POINTS', 'f_110']
    state_test_3D_varypvdDisplay.DiffuseColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.LookupTable = f_110LUT
    state_test_3D_varypvdDisplay.MapScalars = 1
    state_test_3D_varypvdDisplay.InterpolateScalarsBeforeMapping = 1
    state_test_3D_varypvdDisplay.Opacity = 1.0
    state_test_3D_varypvdDisplay.PointSize = 2.0
    state_test_3D_varypvdDisplay.LineWidth = 1.0
    state_test_3D_varypvdDisplay.Interpolation = 'Gouraud'
    state_test_3D_varypvdDisplay.Specular = 0.0
    state_test_3D_varypvdDisplay.SpecularColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.SpecularPower = 100.0
    state_test_3D_varypvdDisplay.Ambient = 0.0
    state_test_3D_varypvdDisplay.Diffuse = 1.0
    state_test_3D_varypvdDisplay.EdgeColor = [0.0, 0.0, 0.5]
    state_test_3D_varypvdDisplay.BackfaceRepresentation = 'Follow Frontface'
    state_test_3D_varypvdDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.BackfaceOpacity = 1.0
    state_test_3D_varypvdDisplay.Position = [0.0, 0.0, 0.0]
    state_test_3D_varypvdDisplay.Scale = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.Orientation = [0.0, 0.0, 0.0]
    state_test_3D_varypvdDisplay.Origin = [0.0, 0.0, 0.0]
    state_test_3D_varypvdDisplay.Pickable = 1
    state_test_3D_varypvdDisplay.Texture = None
    state_test_3D_varypvdDisplay.Triangulate = 0
    state_test_3D_varypvdDisplay.NonlinearSubdivisionLevel = 1
    state_test_3D_varypvdDisplay.UseDataPartitions = 0
    state_test_3D_varypvdDisplay.OSPRayUseScaleArray = 0
    state_test_3D_varypvdDisplay.OSPRayScaleArray = 'f_110'
    state_test_3D_varypvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    state_test_3D_varypvdDisplay.Orient = 0
    state_test_3D_varypvdDisplay.OrientationMode = 'Direction'
    state_test_3D_varypvdDisplay.SelectOrientationVectors = 'None'
    state_test_3D_varypvdDisplay.Scaling = 0
    state_test_3D_varypvdDisplay.ScaleMode = 'No Data Scaling Off'
    state_test_3D_varypvdDisplay.ScaleFactor = 0.6000000000000001
    state_test_3D_varypvdDisplay.SelectScaleArray = 'f_110'
    state_test_3D_varypvdDisplay.GlyphType = 'Arrow'
    state_test_3D_varypvdDisplay.UseGlyphTable = 0
    state_test_3D_varypvdDisplay.GlyphTableIndexArray = 'f_110'
    state_test_3D_varypvdDisplay.UseCompositeGlyphTable = 0
    state_test_3D_varypvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    state_test_3D_varypvdDisplay.SelectionCellLabelBold = 0
    state_test_3D_varypvdDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    state_test_3D_varypvdDisplay.SelectionCellLabelFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.SelectionCellLabelFontSize = 18
    state_test_3D_varypvdDisplay.SelectionCellLabelItalic = 0
    state_test_3D_varypvdDisplay.SelectionCellLabelJustification = 'Left'
    state_test_3D_varypvdDisplay.SelectionCellLabelOpacity = 1.0
    state_test_3D_varypvdDisplay.SelectionCellLabelShadow = 0
    state_test_3D_varypvdDisplay.SelectionPointLabelBold = 0
    state_test_3D_varypvdDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    state_test_3D_varypvdDisplay.SelectionPointLabelFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.SelectionPointLabelFontSize = 18
    state_test_3D_varypvdDisplay.SelectionPointLabelItalic = 0
    state_test_3D_varypvdDisplay.SelectionPointLabelJustification = 'Left'
    state_test_3D_varypvdDisplay.SelectionPointLabelOpacity = 1.0
    state_test_3D_varypvdDisplay.SelectionPointLabelShadow = 0
    state_test_3D_varypvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    state_test_3D_varypvdDisplay.ScalarOpacityFunction = f_110PWF
    state_test_3D_varypvdDisplay.ScalarOpacityUnitDistance = 0.3204642840846154
    state_test_3D_varypvdDisplay.SelectMapper = 'Projected tetra'
    state_test_3D_varypvdDisplay.SamplingDimensions = [128, 128, 128]
    state_test_3D_varypvdDisplay.UseFloatingPointFrameBuffer = 1
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    state_test_3D_varypvdDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Arrow' selected for 'GlyphType'
    state_test_3D_varypvdDisplay.GlyphType.TipResolution = 6
    state_test_3D_varypvdDisplay.GlyphType.TipRadius = 0.1
    state_test_3D_varypvdDisplay.GlyphType.TipLength = 0.35
    state_test_3D_varypvdDisplay.GlyphType.ShaftResolution = 6
    state_test_3D_varypvdDisplay.GlyphType.ShaftRadius = 0.03
    state_test_3D_varypvdDisplay.GlyphType.Invert = 0
    
    # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitle = 'X Axis'
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitle = 'Y Axis'
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitle = 'Z Axis'
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleBold = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleItalic = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleFontSize = 12
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleShadow = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XTitleOpacity = 1.0
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleBold = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleItalic = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleFontSize = 12
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleShadow = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YTitleOpacity = 1.0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleBold = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleItalic = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleFontSize = 12
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleShadow = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZTitleOpacity = 1.0
    state_test_3D_varypvdDisplay.DataAxesGrid.FacesToRender = 63
    state_test_3D_varypvdDisplay.DataAxesGrid.CullBackface = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.CullFrontface = 1
    state_test_3D_varypvdDisplay.DataAxesGrid.GridColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.ShowGrid = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ShowEdges = 1
    state_test_3D_varypvdDisplay.DataAxesGrid.ShowTicks = 1
    state_test_3D_varypvdDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
    state_test_3D_varypvdDisplay.DataAxesGrid.AxesToLabel = 63
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelBold = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelItalic = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelFontSize = 12
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelShadow = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XLabelOpacity = 1.0
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelBold = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelItalic = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelFontSize = 12
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelShadow = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YLabelOpacity = 1.0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelBold = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelItalic = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelFontSize = 12
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelShadow = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZLabelOpacity = 1.0
    state_test_3D_varypvdDisplay.DataAxesGrid.XAxisNotation = 'Mixed'
    state_test_3D_varypvdDisplay.DataAxesGrid.XAxisPrecision = 2
    state_test_3D_varypvdDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.XAxisLabels = []
    state_test_3D_varypvdDisplay.DataAxesGrid.YAxisNotation = 'Mixed'
    state_test_3D_varypvdDisplay.DataAxesGrid.YAxisPrecision = 2
    state_test_3D_varypvdDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.YAxisLabels = []
    state_test_3D_varypvdDisplay.DataAxesGrid.ZAxisNotation = 'Mixed'
    state_test_3D_varypvdDisplay.DataAxesGrid.ZAxisPrecision = 2
    state_test_3D_varypvdDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
    state_test_3D_varypvdDisplay.DataAxesGrid.ZAxisLabels = []
    
    # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    state_test_3D_varypvdDisplay.PolarAxes.Visibility = 0
    state_test_3D_varypvdDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
    state_test_3D_varypvdDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
    state_test_3D_varypvdDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
    state_test_3D_varypvdDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.EnableCustomRange = 0
    state_test_3D_varypvdDisplay.PolarAxes.CustomRange = [0.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.RadialAxesVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.DrawRadialGridlines = 1
    state_test_3D_varypvdDisplay.PolarAxes.PolarArcsVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.DrawPolarArcsGridlines = 1
    state_test_3D_varypvdDisplay.PolarAxes.NumberOfRadialAxes = 0
    state_test_3D_varypvdDisplay.PolarAxes.AutoSubdividePolarAxis = 1
    state_test_3D_varypvdDisplay.PolarAxes.NumberOfPolarAxis = 0
    state_test_3D_varypvdDisplay.PolarAxes.MinimumRadius = 0.0
    state_test_3D_varypvdDisplay.PolarAxes.MinimumAngle = 0.0
    state_test_3D_varypvdDisplay.PolarAxes.MaximumAngle = 90.0
    state_test_3D_varypvdDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
    state_test_3D_varypvdDisplay.PolarAxes.Ratio = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitle = 'Radial Distance'
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleLocation = 'Bottom'
    state_test_3D_varypvdDisplay.PolarAxes.PolarLabelVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.PolarLabelFormat = '%-#6.3g'
    state_test_3D_varypvdDisplay.PolarAxes.PolarLabelExponentLocation = 'Labels'
    state_test_3D_varypvdDisplay.PolarAxes.RadialLabelVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.RadialLabelFormat = '%-#3.1f'
    state_test_3D_varypvdDisplay.PolarAxes.RadialLabelLocation = 'Bottom'
    state_test_3D_varypvdDisplay.PolarAxes.RadialUnitsVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.ScreenSize = 10.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleBold = 0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleItalic = 0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleShadow = 0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTitleFontSize = 12
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelBold = 0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelItalic = 0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelShadow = 0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisLabelFontSize = 12
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextBold = 0
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextItalic = 0
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextShadow = 0
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextColor = [1.0, 1.0, 1.0]
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
    state_test_3D_varypvdDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
    state_test_3D_varypvdDisplay.PolarAxes.EnableDistanceLOD = 1
    state_test_3D_varypvdDisplay.PolarAxes.DistanceLODThreshold = 0.7
    state_test_3D_varypvdDisplay.PolarAxes.EnableViewAngleLOD = 1
    state_test_3D_varypvdDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
    state_test_3D_varypvdDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
    state_test_3D_varypvdDisplay.PolarAxes.PolarTicksVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
    state_test_3D_varypvdDisplay.PolarAxes.TickLocation = 'Both'
    state_test_3D_varypvdDisplay.PolarAxes.AxisTickVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.AxisMinorTickVisibility = 0
    state_test_3D_varypvdDisplay.PolarAxes.ArcTickVisibility = 1
    state_test_3D_varypvdDisplay.PolarAxes.ArcMinorTickVisibility = 0
    state_test_3D_varypvdDisplay.PolarAxes.DeltaAngleMajor = 10.0
    state_test_3D_varypvdDisplay.PolarAxes.DeltaAngleMinor = 5.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
    state_test_3D_varypvdDisplay.PolarAxes.ArcMajorTickSize = 0.0
    state_test_3D_varypvdDisplay.PolarAxes.ArcTickRatioSize = 0.3
    state_test_3D_varypvdDisplay.PolarAxes.ArcMajorTickThickness = 1.0
    state_test_3D_varypvdDisplay.PolarAxes.ArcTickRatioThickness = 0.5
    state_test_3D_varypvdDisplay.PolarAxes.Use2DMode = 0
    state_test_3D_varypvdDisplay.PolarAxes.UseLogAxis = 0
    
    # reset view to fit data
    renderView1.ResetCamera()
    
    # show color bar/color legend
    state_test_3D_varypvdDisplay.SetScalarBarVisibility(renderView1, True)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # get color legend/bar for f_110LUT in view renderView1
    f_110LUTColorBar = GetScalarBar(f_110LUT, renderView1)
    
    # change scalar bar placement
    f_110LUTColorBar.WindowLocation = 'AnyLocation'
    f_110LUTColorBar.Position = [0.8098495212038302, 0.43956043956043955]
    f_110LUTColorBar.ScalarBarLength = 0.32999999999999996
    
    # change scalar bar placement
    f_110LUTColorBar.Position = [0.8098495212038302, 0.22893772893772893]
    f_110LUTColorBar.ScalarBarLength = 0.5406227106227105
    
    # current camera placement for renderView1
    renderView1.CameraPosition = [3.0, 2.0, 14.214227679878107]
    renderView1.CameraFocalPoint = [3.0, 2.0, 0.25]
    renderView1.CameraParallelScale = 3.61420807370024
    
    # save screenshot
    SaveScreenshot('/home/hwan/Downloads/test.png', renderView1, ImageResolution=[731, 546],
        FontScaling='Scale fonts proportionally',
        OverrideColorPalette='',
        StereoMode='No change',
        TransparentBackground=1,
        ImageQuality=100)
    
    #### saving camera placements for all active views
    
    # current camera placement for renderView1
    renderView1.CameraPosition = [3.0, 2.0, 14.214227679878107]
    renderView1.CameraFocalPoint = [3.0, 2.0, 0.25]
    renderView1.CameraParallelScale = 3.61420807370024
    
    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).