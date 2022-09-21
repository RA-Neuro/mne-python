"""York Instruments Constants"""

# Authors: Richard Aveyard <Richard.A.Aveyard@gmail.com>
#          Nick Greenall <nick.greenall@york-instruments.co.uk>
# License: BSD (3-clause)

from os import path

from ...utils import BunchConst

YI = BunchConst()

# HDF5 paths
YI.PATH_ACQUISITIONS = '/acquisitions'
YI.PATH_GEOMETRY = '/geometry'
YI.PATH_CONFIG = '/config'
YI.PATH_SUBJECT = '/subject'
YI.PATH_APPS = '/apps'
YI.PATH_NOTES = '/notes'

YI.PATH_DEFAULT_ACQUISITION = path.join(YI.PATH_ACQUISITIONS, 'default')
YI.PATH_TEMPLATE_ACQUISITION = path.join(YI.PATH_ACQUISITIONS, '{}') #  use with format
YI.REL_PATH_ACQUISITION_EPOCHS = 'epochs' # relative to acquisition path
YI.REL_PATH_ACQUISITION_FITTED_COILS = 'fitted_coils'

YI.PATH_COIL_GEOMETRY = path.join(YI.PATH_GEOMETRY, 'coils')
YI.PATH_FIDUCIAL_GEOMETRY = path.join(YI.PATH_GEOMETRY, 'fiducials')
YI.PATH_HEAD_GEOMETRY = path.join(YI.PATH_GEOMETRY, 'head_shape')

YI.PATH_CHANNEL_CONFIG = path.join(YI.PATH_CONFIG, 'channels')
YI.PATH_WEIGHTS = path.join(YI.PATH_CONFIG, 'weights')
YI.PATH_PROTOCOL = path.join(YI.PATH_CONFIG, 'protocol')

# YI app path examples
YI.PATH_VENDOR_YI = path.join(YI.PATH_APPS, 'york-instruments.co.uk')
YI.PATH_DATA_VIEWER_APP = path.join(YI.PATH_VENDOR_YI, 'data_viewer')

# Unit strings
YI.UNIT_T = 'T'  # Tesla
YI.UNIT_M = 'm'  # metre
YI.UNIT_A = 'A'  # Ampere
YI.UNIT_V = 'V'  # Volts
YI.UNIT_b = 'b'  # bits

# and multipliers
YI.UNIT_MULT_F = 'f'  # e-15
YI.UNIT_MULT_PI = 'p'  # e-12
YI.UNIT_MULT_N = 'n'  # e-9
YI.UNIT_MULT_U = 'μ'  # e-6
YI.UNIT_MULT_MI = 'm'  # e-3
YI.UNIT_MULT_K = 'k'  # e+3
YI.UNIT_MULT_ME = 'M'  # e+6
YI.UNIT_MULT_G = 'G'  # e+9
YI.UNIT_MULT_T = 'T'  # e+12
YI.UNIT_MULT_PE = 'P'  # e+15

# Coil types
YI.COIL_MAG = 'magnetometer'  # TODO update to reflect new format revision
YI.COIL_AXIAL_GRAD = 'axial gradiometer'
YI.COIL_PLANAR_GRAD = 'planar gradiometer'
YI.COIL_SHORTED = 'shorted coil'
YI.COIL_OPEN = 'open coil'

# YI Channel Types
YI.CHAN_TYPE_ANALOG = 'ANALOG'
YI.CHAN_TYPE_COH = 'COH'
YI.CHAN_TYPE_DIGITAL = 'DIGITAL'
YI.CHAN_TYPE_EEG = 'EEG'
YI.CHAN_TYPE_MEG = 'MEG'
YI.CHAN_TYPE_MEGREF = 'MEGREF'  # TODO remove when redundent
YI.CHAN_TYPE_TEMP = 'TEMPERATURE'

# YI Channel Mode
YI.CHAN_MODE_TRG = 'TRIGGER'
YI.CHAN_MODE_RSP = 'RESPONSE'
YI.CHAN_MODE_SYNC_IN = 'SYNC_IN'
YI.CHAN_MODE_SYNC_OUT = 'SYNC_OUT'
YI.CHAN_MODE_BIN = 'BIN'

