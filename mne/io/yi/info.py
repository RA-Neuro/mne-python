# Read info from York Instruments HDF5 file

# Authors: Richard Aveyard <Richard.A.Aveyard@gmail.com>
#          Nick Greenall <nick.greenall@york-instruments.co.uk>
# License: BSD (3-clause)

import os.path as op
import re

import numpy as np

from ..meas_info import _empty_info
from .constants import YI
from ..constants import FIFF
from ...transforms import Transform
from datetime import datetime, timezone
from .._digitization import _make_dig_points

YI_TO_FIFF_UNITS = {
    YI.UNIT_T: FIFF.FIFF_UNIT_T,
    YI.UNIT_M: FIFF.FIFF_UNIT_M,
    YI.UNIT_A: FIFF.FIFF_UNIT_A,
    YI.UNIT_V: FIFF.FIFF_UNIT_V,
    YI.UNIT_b: FIFF.FIFF_UNIT_V,
}
YI_TO_FIFF_UNIT_MULT = {
    YI.UNIT_MULT_F: FIFF.FIFF_UNITM_F,
    YI.UNIT_MULT_PI: FIFF.FIFF_UNITM_P,
    YI.UNIT_MULT_N: FIFF.FIFF_UNITM_N,
    YI.UNIT_MULT_U: FIFF.FIFF_UNITM_MU,
    YI.UNIT_MULT_MI: FIFF.FIFF_UNITM_M,
    YI.UNIT_MULT_K: FIFF.FIFF_UNITM_K,
    YI.UNIT_MULT_ME: FIFF.FIFF_UNITM_MEG,
    YI.UNIT_MULT_G: FIFF.FIFF_UNITM_GIG,
    YI.UNIT_MULT_T: FIFF.FIFF_UNITM_T,
    YI.UNIT_MULT_PE: FIFF.FIFF_UNITM_PET,
}
YI_TO_FIFF_CHAN_TYPE = {
    YI.CHAN_TYPE_ANALOG: FIFF.FIFFV_MISC_CH,
    YI.CHAN_TYPE_DIGITAL: FIFF.FIFFV_MISC_CH,
    YI.CHAN_TYPE_TEMP: FIFF.FIFFV_SYST_CH,
    YI.CHAN_TYPE_COH: FIFF.FIFFV_SYST_CH,
    YI.CHAN_TYPE_EEG: FIFF.FIFFV_EEG_CH,
    YI.CHAN_TYPE_MEG: FIFF.FIFFV_MEG_CH,
    YI.CHAN_TYPE_MEGREF: FIFF.FIFFV_MEG_CH,  # TODO update
}

unit_pat = '|'.join((
    val for const, val in YI.items()
    if val in YI_TO_FIFF_UNITS and re.search('UNIT_(?!MULT).*', const)
))  # ~ 'T|m|A|...|V', allows multichar unit strings

mult_pat = '|'.join((
    val for const, val in YI.items()
    if val in YI_TO_FIFF_UNIT_MULT and re.search('UNIT_MULT.*', const)
))  # ~ 'f|m|n|...|P', allows multichar unit multiplier strings

UNIT_RE = re.compile(
    r'^\s*'  # Ignore surrounding whitespace
    '(?P<mult>({mult_pat})'  # Optionally find unit multiplier string
    '(?={unit_pat}))?'  # but only if followed by unit
    '(?P<unit>{unit_pat})?'  # Optionally find a unit string
    '\s*$'.format(mult_pat=mult_pat, unit_pat=unit_pat)
)


def _convert_channel_pos_ori(pos, ori, coil_type):
    """
    Converts a YI channel position and orientation arrays to
    a MNE compatible loc array.

    The position and orientation arrays define 3-element vectors
    for each loop in the device (currently between 1 or 2). 
    The orientation vector defines the normal vector (right hand
    rule) of the loop. Axial rotation around this vector is 
    not defined by the orientation vector (due to rotational
    symmetry of loops).

    This function uses a YI convention for the position and orientation 
    of the device: the first coil is the *primary* coil and the
    second coil's position and orientation is derived from this
    primary. Therefore the devices location and orientation is
    effectively defined by the first coil (closest to head).

    For magnetometers and axial gradiometers:
        
        loc[:3] = pos[0]
        loc[3:] = rotation matrix formed from orientation vector.

        The rotation matrix is formed by the assumption that the
        device is rotated from pointing in the positive z-direction
        to point in the orientation direction. Thus:

              | ori[0] x [0, 0, 1]           |
        rot = | ori[0] x (ori[0] x [0, 0, 1] |
              | ori[0]                       |

        **The axial rotation around the orientation vector is thus
        not defined.**

    For planar gradiometers: NOT IMPLEMENTED
        
        *probably* 
        loc[:3] = np.mean(pos[0], pos[1])

              | ori[0] x (pos[0] - pos[1]) |
        rot = | pos[0] - pos[1]            |
              | ori[0]                     |

    Args:
        pos (numpy.array/h5py.dataset) : Nx3 loop position vectors
        ori (numpy.array/h5py.dataset) : Nx3 loop orientation (normal)
                                         vectors
        coil_type (str) : YI constant describing device type

    Returns:
        loc (numpy.array) : 12 element array containing position and 
                            3x3 rotation matrix for orientation.

    Raises:
        NotImplementedError : if coil_type == 'planar gradiometer'
        ValueError : if coil_type is unsupported.

    """
    if len(pos) > 2 and len(pos) < 1 and len(ori) != len(pos):
        ValueError(
            'Expected either a single coil or 2 coil device,'
            ' received a {} positions and {} orientations.'.format(
                len(pos), len(ori)
            )
        )
    if coil_type == YI.COIL_MAG or coil_type == YI.COIL_AXIAL_GRAD:
        if np.all(ori[0] == [0, 0, 1]):
            ori_othog_1 = np.array([1, 0, 0])
            ori_othog_2 = np.array([0, 1, 0])
        else:
            ori_othog_1 = np.cross(ori[0], [0, 0, 1])
            ori_othog_1 = ori_othog_1 / np.linalg.norm(ori_othog_1)
            ori_othog_2 = np.cross(ori[0], ori_othog_1)
            ori_othog_2 = ori_othog_2 / np.linalg.norm(ori_othog_2)
        rot = np.vstack((
            ori_othog_1,
            ori_othog_2,
            ori[0],
        ))
        pos_p=pos[0]
        pos_p /= 1000.
        return np.concatenate((
            pos_p,
            rot.flatten(),
        ))
    elif coil_type == YI.COIL_PLANAR_GRAD:
        NotImplementedError(
            'Planar gradiometers currently unsupported'
        )
    else:
        ValueError(
            'Unrecognised coil type {}'.format(coil_type)
        )


def _convert_channel_unit(unit_str):
    """
    Converts a YI channel unit string to a compatible FIFF unit
    and multiplier code.

    Args:
        unit_str (str): YI unit string (found in hdf5 files) e.g. "μV"

    Returns:
        dict: dictionary containing the keys "unit" and "unit_mul",
              with the corresponding correct FIFF unit and multiplier
              codes

    Raises:
        ValueError: If the unit_str is not recognised or FIFF
                    incompatible
    """
    unit_mtch = UNIT_RE.search(unit_str)
    if unit_mtch:
        unit = YI_TO_FIFF_UNITS.get(
            unit_mtch.group('unit'),
            FIFF.FIFF_UNIT_NONE
        )
        mult = YI_TO_FIFF_UNIT_MULT.get(
            unit_mtch.group('mult'),
            FIFF.FIFF_UNITM_NONE
        )
    else:
        raise ValueError(
            'Unit string "{}" in YI HDF5 file is not recognised or'
            ' is not convertible to FIFF unit'.format(unit_str)
        )

    return dict(
        unit=unit,
        unit_mul=mult,
    )


def _convert_channel_config(chan_config, upb_applied=False):
    """
    Converts a YI channel config mapping to a MNE chs dict

    Args:
        chan_config (h5py.group) : Channel configuration

    Kwargs:
        upb_applied (Boolean) : has the calibration scaling been
                                pre-applied to data

    Returns:
        dict : Dictionary which is w compatible with MNE Info['chs']
    """

    if chan_config.attrs['chan_type']=='MEG':
        coil_T=FIFF.FIFFV_COIL_MAGNES_MAG
        conv_loc = _convert_channel_pos_ori(
            chan_config['position'],
            chan_config['orientation'],
            YI.COIL_MAG,)
        chan_K = FIFF.FIFFV_MEG_CH
    elif chan_config.attrs['chan_type']=='MEGREF':
        chan_K = FIFF.FIFFV_REF_MEG_CH
        if chan_config.attrs['io_name'].startswith('M'):
            coil_T=FIFF.FIFFV_COIL_MAGNES_REF_MAG
            conv_loc = _convert_channel_pos_ori(
            chan_config['position'],
            chan_config['orientation'],
            YI.COIL_MAG,)
        elif chan_config.attrs['io_name'].startswith('G'):
            if chan_config.attrs['io_name'] in ('GxxA', 'GyyA'):
                coil_T=FIFF.FIFFV_COIL_MAGNES_REF_GRAD
                conv_loc = _convert_channel_pos_ori(
                chan_config['position'],
                chan_config['orientation'],
                YI.COIL_MAG,)
            elif chan_config.attrs['io_name'] in ('GyXA', 'GzxA', 'GzyA'):
                coil_T=FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD  
                conv_loc = _convert_channel_pos_ori(
                chan_config['position'],
                chan_config['orientation'],
                YI.COIL_MAG,)
        elif chan_config.attrs['io_name'] in ('SA1', 'SA2', 'SA3'):
            conv_loc = np.full(12, np.nan)
            coil_T = FIFF.FIFFV_COIL_NONE
            chan_K = FIFF.FIFFV_MISC_CH
    elif chan_config.attrs['chan_type']=='DIGITAL':
        if chan_config.attrs['io_name'] in ('P_PORT_A', 'P_PORT_B', 'BNC_1-8'):
            conv_loc = np.full(12, np.nan)
            coil_T = FIFF.FIFFV_COIL_NONE
            chan_K = FIFF.FIFFV_STIM_CH
        elif chan_config.attrs['io_name'] in ('R_BOX1','R_BOX2'):
            conv_loc = np.full(12, np.nan)
            coil_T = FIFF.FIFFV_COIL_NONE
            chan_K = FIFF.FIFFV_STIM_CH
        else:
            conv_loc = np.full(12, np.nan)
            coil_T = FIFF.FIFFV_COIL_NONE
            chan_K = FIFF.FIFFV_MISC_CH
    else:
        conv_loc = np.full(12, np.nan)
        coil_T = FIFF.FIFFV_COIL_NONE
        chan_K = FIFF.FIFFV_MISC_CH
    return dict(
        cal=chan_config.attrs['units_per_bit'] if upb_applied else 1,
        range=1 / chan_config.attrs.get('gain', 1),
        logno=1,
        scanno=1,
        coord_frame=FIFF.FIFFV_COORD_DEVICE,
        coil_type=coil_T,  
        loc=conv_loc,
        ch_name=op.basename(chan_config.name),
        kind=chan_K,
        **_convert_channel_unit(chan_config.attrs['units']),
    )


def _compose_meas_info(hdf5_file, acquisition=None):
    """
    Create measurement information from YI HDF5 file

    Args:
        hdf5_file (h5py.File) : HDF5 file in York Instruments format

    KWArgs:
        acquisition (h5py.group) : Acquisition in HDF5 file. This can
            be used to define additional, per acquisition information

    Returns:
        mne.Info : measurement information

    """

    info = _empty_info(
        hdf5_file[YI.PATH_DEFAULT_ACQUISITION].attrs['sample_rate']
    )

    info['line_freq'] = hdf5_file[YI.PATH_CONFIG].attrs['supply_freq']
    if acquisition is None:
        info['ch_names'] = list(hdf5_file[YI.PATH_CHANNEL_CONFIG].keys())
        info['chs'] = [
            _convert_channel_config(chan)
            for chan in hdf5_file[YI.PATH_CHANNEL_CONFIG].values()
        ]
    else:
        acquisition_ch_names = acquisition['channel_list']
        info['ch_names'] = list(acquisition_ch_names)
        info['chs'] = [
            _convert_channel_config(hdf5_file[op.join(
                YI.PATH_CHANNEL_CONFIG,
                chan,
            )]) for chan in acquisition_ch_names
        ]
    info['nchan'] = len(info['chs'])
    try:
        trans = acquisition['ccs_to_scs_transform'][...]
        trans[0:3,3] /=1000. # mm->m conversion for translations in affine transform
        info['dev_head_t'] = Transform('meg', 'head', trans)
        all_points = hdf5_file[YI.PATH_HEAD_GEOMETRY]['head_shape'][...]
        all_points /=1000. # mm->m conversion
        nas_p = hdf5_file[YI.PATH_FIDUCIAL_GEOMETRY]['nas/location'][...].squeeze()
        nas_p /= 1000.
        lpa_p = hdf5_file[YI.PATH_FIDUCIAL_GEOMETRY]['lpa/location'][...].squeeze()
        lpa_p /= 1000.
        rpa_p = hdf5_file[YI.PATH_FIDUCIAL_GEOMETRY]['rpa/location'][...].squeeze()
        rpa_p /= 1000.
        hpi_p = np.zeros([5,3])
        for coil in hdf5_file[YI.PATH_COIL_GEOMETRY]:
            hpi_p[int(coil)-1,:] = hdf5_file[YI.PATH_COIL_GEOMETRY][str(coil)+'/location'][...][0,:]
        hpi_p /= 1000.
        info['dig'] = _make_dig_points(nasion=nas_p,lpa=lpa_p,rpa=rpa_p, hpi = hpi_p,extra_points=all_points,)
    except:
        trans = np.identity(4)
        print('\n WARNING: NO HEAD GEOMETRY FOUND!\n Proceeding with identity transform. \n')

    start_date_time = datetime.strptime(acquisition.attrs['start_time'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
    info['meas_date'] = start_date_time.replace(tzinfo=timezone.utc)
    return info
