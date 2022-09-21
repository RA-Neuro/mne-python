# Author: Nicholas Greenall <nick.greenall@york-instruments.co.uk>
#
# License: BSD (3-clause)

import os
import h5py
import pytest

import numpy as np

from mne.io.yi.yi import (
    RawYI,
)

YI_PATH_TEST_DATA = os.environ['YI_TEST_PATH']
YI_PATH_4_CHAN_RAND_SYNTH_HDF5 = os.path.join(
    YI_PATH_TEST_DATA,
    '4-channel-hash-synth.hdf5'
)

RANDOM_SEED = 32
RANDOM_DATA_SHAPE = (2000, 4)


def gen_random_synthetic_data():
    """
    This function generates random data of fixed size from
    RANDOM_SEED. This data should be identical to that stored
    in 4-channel-hash-synth.hdf5. Running a hash on this and
    that dataset should be identical.
    """
    random_state = np.random.RandomState(seed=RANDOM_SEED)
    return random_state.randn(*RANDOM_DATA_SHAPE)


def test_synthetic_data_hash():
    raw = RawYI(YI_PATH_4_CHAN_RAND_SYNTH_HDF5, preload=True)
    comp_data = gen_random_synthetic_data()
    assert hash(raw._data.T.tostring()) == hash(comp_data.tostring())


def test_invalid_acquisition_hdf5(tmp_path):
    invalid_file_path = tmp_path / 'invalid.hdf5'
    with h5py.File(invalid_file_path, 'w') as f:
        f.attrs['foo'] = 'bar'  # filler
    with pytest.raises(KeyError, match='.*not find acquisition "default".*'):
        raw = RawYI(invalid_file_path)  # noqa
