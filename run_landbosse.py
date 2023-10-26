from main_function import run_landbosse
import numpy as np

# **** Inputs ****
# 1. (Relative) Turbine Coordinates in [km]
# Make sure that number of turbine coordinates match with number of turbines in evaluated project!
Turbine_coordinates = np.array([(3.50,4.99),(3.70,3.20),(1.10,3.57),(3.37,3.58),(2.34,4.50),(2.53,2.01),(4.28,4.61),(1.60,3.73),(3.50,4.23),(4.39,0.68),(0.16,4.35),(3.23,3.75),(0.54,0.57),(4.85,1.96),(0.99,2.34)])
#
# 2. (Relative) Substation Coordinate in [km]
# Make sure to place the substation towards middle of plant (due to minimum spanning tree)
Substation_coordinate = np.array([(2.5,2.5)])
#
# 3. Cable Voltage in [kV]
# Possible values: 10,20,30,45,60 kV
Cable_Voltage = 60

# **** Call LandBOSSE ****
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
if __name__ == '__main__':
    from multiprocessing import freeze_support  # Import freeze_support
    freeze_support()  # Call freeze_support
    exit_code = run_landbosse(Turbine_coordinates, Substation_coordinate,Cable_Voltage)
    exit(exit_code)