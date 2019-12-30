import sys
sys.path.append('../..')

from paraview.simple import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Plot and Save Parameter and State                     #
###############################################################################
def plot_and_save_predictions_paraview(file_paths, cbar_RGB_parameter_test, cbar_RGB_state_test):
    #=== Parameter Test ===#
    pvd_load_filepath = file_paths.figures_savefile_name_parameter_test + '.pvd'
    figure_save_filepath = file_paths.figures_savefile_name_parameter_test + '.png'
    save_png(pvd_load_filepath, figure_save_filepath, cbar_RGB_parameter_test)
    
    #=== State Test ===#
    pvd_load_filepath = file_paths.figures_savefile_name_state_test + '.pvd'
    figure_save_filepath = file_paths.figures_savefile_name_state_test + '.png'
    save_png(pvd_load_filepath, figure_save_filepath, cbar_RGB_state_test)
    
    #=== Parameter Predictions ===#
    pvd_load_filepath = file_paths.figures_savefile_name_parameter_pred + '.pvd'
    figure_save_filepath = file_paths.figures_savefile_name_parameter_pred + '.png'
    save_png(pvd_load_filepath, figure_save_filepath, cbar_RGB_parameter_test)
    
    #=== State Predictions ===#
    pvd_load_filepath = file_paths.figures_savefile_name_state_pred + '.pvd'
    figure_save_filepath = file_paths.figures_savefile_name_state_pred+ '.png'
    save_png(pvd_load_filepath, figure_save_filepath, cbar_RGB_state_test)

###############################################################################
#                             Save Paraview Plot                              #
###############################################################################
def save_png(pvd_load_filepath, figure_save_filepath, cbar_RGB):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    parameter_or_state_pvd = PVDReader(FileName=pvd_load_filepath)
    
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.OrientationAxesVisibility = 0
    
    # show data in view
    parameter_or_state_pvdDisplay = Show(parameter_or_state_pvd, renderView1)
    
    #=== Colourbar ===#
    f_110LUT = GetColorTransferFunction('f_110')
    f_110LUT.RGBPoints = cbar_RGB
    f_110LUT.ScalarRangeInitialized = 1.0
    f_110LUTColorBar = GetScalarBar(f_110LUT, renderView1)
    f_110LUTColorBar.Title = ''
    f_110LUTColorBar.ComponentTitle = ''
    f_110LUTColorBar.WindowLocation = 'AnyLocation'
    f_110LUTColorBar.Position = [0.7332402234636871, 0.21437578814627997]
    f_110LUTColorBar.ScalarBarLength = 0.5582471626733922
    f_110LUTColorBar.ScalarBarThickness = 5
    f_110LUTColorBar.LabelFontSize = 5
    f_110LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    
    # current camera placement for renderView1
    renderView1.CameraPosition = [3.0, 2.0, 14.214227679878107]
    renderView1.CameraFocalPoint = [3.0, 2.0, 0.25]
    renderView1.CameraParallelScale = 3.61420807370024
    
    # save screenshot
    SaveScreenshot(figure_save_filepath, renderView1, ImageResolution=[1432, 793], TransparentBackground=1)