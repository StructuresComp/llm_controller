import numpy as np
import Discrete_Elastic_Beams as deb

def getConfiguration(boundary_conditions):
    message = "Configuration is valid"
    shape = boundary_conditions.shape
    if len(shape) != 2 or shape[1] != 2:
        message = "Configuration should be a 2D array with 2 columns (x and y coordinates)."
        return [], message
    config = deb.discreteBeam()
    print(config)

    return config, message