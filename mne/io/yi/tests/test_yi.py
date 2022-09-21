# Author: Nicholas Greenall <nick.greenall@york-instruments.co.uk>
#
# License: BSD (3-clause)

import h5py
import os
import pytest

import numpy as np

from mne.io.meas_info import Info
from mne.io.yi.info import (
    _compose_meas_info,
    _convert_channel_pos_ori,
    _convert_channel_unit,
)
from mne.io.yi.constants import YI
from mne.io.constants import FIFF

YI_PATH_TEST_DATA = os.environ['YI_TEST_PATH']
YI_PATH_1_CHAN_SYNTH_HDF5 = os.path.join(
    YI_PATH_TEST_DATA,
    '1-channel-synth.hdf5'
)


class TestUnitConv(object):

    def test_invalid_unit_exception(self):
        with pytest.raises(ValueError, match=r'^Unit string "foo".*'):
            _convert_channel_unit('foo')

    def test_no_unit_conv(self):
        unit_dict = _convert_channel_unit('')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_NONE
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_NONE

    def test_m_unit_conv(self):
        unit_dict = _convert_channel_unit('m')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_M
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_NONE

    def test_A_unit_conv(self):
        unit_dict = _convert_channel_unit('A')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_A
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_NONE

    def test_V_unit_conv(self):
        unit_dict = _convert_channel_unit('V')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_V
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_NONE

    def test_T_conv(self):
        unit_dict = _convert_channel_unit('T')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_NONE

    def test_f_mult_conv(self):
        unit_dict = _convert_channel_unit('fT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_F

    def test_p_mult_conv(self):
        unit_dict = _convert_channel_unit('pT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_P

    def test_n_mult_conv(self):
        unit_dict = _convert_channel_unit('nT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_N

    def test_μ_mult_conv(self):
        unit_dict = _convert_channel_unit('μT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_MU

    def test_m_mult_conv(self):
        unit_dict = _convert_channel_unit('mT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_M

    def test_k_mult_conv(self):
        unit_dict = _convert_channel_unit('kT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_K

    def test_M_mult_conv(self):
        unit_dict = _convert_channel_unit('MT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_MEG

    def test_G_mult_conv(self):
        unit_dict = _convert_channel_unit('GT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_GIG

    def test_T_mult_conv(self):
        unit_dict = _convert_channel_unit('TT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_T

    def test_P_mult_conv(self):
        unit_dict = _convert_channel_unit('PT')
        assert unit_dict['unit'] == FIFF.FIFF_UNIT_T
        assert unit_dict['unit_mul'] == FIFF.FIFF_UNITM_PET


class TestPosOriConv(object):

    def test_origin_no_ori_rot(self):
        pos = np.array([
            [0, 0, 0],
            [1, 1, 1],
        ])
        ori = np.array([
            [0, 0, 1],
            [0, 0, -1],
        ])
        loc = _convert_channel_pos_ori(pos, ori, YI.COIL_AXIAL_GRAD)

        assert np.all(loc == [
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ])

    def test_pos_pos_no_ori_rot(self):
        pos = np.array([
            [1, 1, 1],
            [1, 1, 1],
        ])
        ori = np.array([
            [0, 0, 1],
            [0, 0, -1],
        ])
        loc = _convert_channel_pos_ori(pos, ori, YI.COIL_AXIAL_GRAD)

        assert np.all(loc == [
            1, 1, 1,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ])

    def test_ori_rot(self):
        pos = np.array([
            [0, 0, 0],
        ])
        ori = np.array([
            [1, 1, 1],
        ])
        ori = ori / np.linalg.norm(ori)
        loc = _convert_channel_pos_ori(pos, ori, YI.COIL_MAG)

        rot = loc[3:].reshape(3, 3)

        # define an plane parallel to XY
        a = np.array([3, -3, -23])
        b = np.array([2, 5, -23])
        c = np.array([1, -1, -23])

        norm_abc = np.cross(a - c, b - c)
        norm_abc = norm_abc / np.linalg.norm(norm_abc)

        assert np.all(norm_abc == [0, 0, 1])  # make sure test is correct

        ra = rot@a
        rb = rot@b
        rc = rot@c

        norm_rabc = np.cross(ra - rc, rb - rc)
        norm_rabc = norm_rabc / np.linalg.norm(norm_rabc)

        # Assert that the plane defined above when rotated has a normal
        # vector which is equal to the ori vector we started with
        # Since we start with an orientation vector - axial rotation is
        # undefined and should not be tested.
        assert np.allclose(norm_rabc, ori)

    def test_ori_rot_mag_invariance(self):
        pos = np.array([
            [0, 0, 0],
        ])
        ori = np.array([
            [1, 1, 1],
        ])
        ori = ori / np.linalg.norm(ori)  # ori must be unit
        loc = _convert_channel_pos_ori(pos, ori, YI.COIL_MAG)

        rot = loc[3:].reshape(3, 3)

        # define an plane parallel to XY
        a = np.array([4, -20, 100])

        ra = rot@a

        assert np.isclose(
            np.linalg.norm(a),
            np.linalg.norm(ra),
        )


class TestSynthInfo(object):

    @pytest.fixture(scope='class')
    def synth_info(self):
        with h5py.File(YI_PATH_1_CHAN_SYNTH_HDF5, 'r') as f:
            info = _compose_meas_info(f)
        return info

    def test_info_type(self, synth_info):
        assert isinstance(synth_info, Info)

    def test_info_sfreq_is_500(self, synth_info):
        assert synth_info['sfreq'] == 500

    def test_info_line_freq_is_50(self, synth_info):
        assert synth_info['line_freq'] == 50.

    def test_info_single_ch_name(self, synth_info):
        assert len(synth_info['ch_names']) == 1

    def test_info_ch_name_is_meg001(self, synth_info):
        assert synth_info['ch_names'][0] == 'MEG001'

    def test_info_chs_singular(self, synth_info):
        assert len(synth_info['chs']) == 1

    def test_info_chs_ch_name_is_meg001(self, synth_info):
        assert synth_info['chs'][0]['ch_name'] == 'MEG001'

    def test_info_chs_cal_is_1(self, synth_info):
        assert synth_info['chs'][0]['cal'] == 1

    def test_info_chs_range_is_1(self, synth_info):
        assert synth_info['chs'][0]['range'] == 1

    def test_info_chs_loc_is_basis(self, synth_info):
        assert np.all(synth_info['chs'][0]['loc'] == np.array([
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]))

    def test_info_chs_gain_is_1(self, synth_info):
        assert synth_info['chs'][0]['range'] == 1

    def test_info_chs_logno_is_1(self, synth_info):
        assert synth_info['chs'][0]['logno'] == 1

    def test_info_chs_scanno_is_1(self, synth_info):
        assert synth_info['chs'][0]['scanno'] == 1

    def test_info_chs_coord_frame_is_device(self, synth_info):
        assert synth_info['chs'][0]['coord_frame'] == FIFF.FIFFV_COORD_DEVICE

    def test_info_chs_coil_type_is_mag(self, synth_info):
        assert synth_info['chs'][0]['coil_type'] == FIFF.FIFFV_COIL_YI_MAG

    def test_info_chs_kind_is_meg(self, synth_info):
        assert synth_info['chs'][0]['kind'] == FIFF.FIFFV_MEG_CH

    def test_info_chs_unit_is_tesla(self, synth_info):
        assert synth_info['chs'][0]['unit'] == FIFF.FIFF_UNIT_T

    def test_info_chs_unit_mul_is_femto(self, synth_info):
        assert synth_info['chs'][0]['unit_mul'] == FIFF.FIFF_UNITM_F
