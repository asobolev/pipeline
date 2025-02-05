# include modules to the path
import sys, os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'session'))
sys.path.append(os.path.join(parent_dir, 'postprocessing'))

import os
import numpy as np
import h5py, json
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

#import scipy.ndimage as ndi

from session.utils import *
from session.adapters import H5NAMES, EPOCH_NAMES, COLORS
from postprocessing.spatial import place_field_2D, map_stats, get_field_patches, get_positions_relative_to
from postprocessing.spatial import bins2meters, cart2pol, pol2cart
from postprocessing.spiketrain import instantaneous_rate

pwd = os.getcwd()
if pwd.startswith('/mnt'):
    # GPU-XXL
    source = '/mnt/nevermind.data-share/ag-grothe/Andrey/data'
    report = '/mnt/nevermind.data-share/ag-grothe/Andrey/analysis'
else:
    # prodesk in the lab
    source = '/home/sobolev/nevermind/Andrey/data'
    report = '/home/sobolev/nevermind/Andrey/analysis/'
