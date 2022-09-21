"""Conversion tool from YI HDF5 to FIF"""

# Authors: Richard Aveyard <Richard.A.Aveyard@gmail.com>
#          Nick Greenall <nick.greenall@york-instruments.co.uk>
# License: BSD (3-clause)


import h5py
import numpy as np
import os.path as path

from ..base import BaseRaw
from ...utils import verbose, fill_doc
from .info import _compose_meas_info
from .constants import YI

@fill_doc
def read_yi_hdf5(hdf5_filename, acquisition_name, preload=False, verbose=None):
    """Raw object from YI hdf5 file

    Parameters
    ----------
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    %(verbose)s

    Returns
    -------
    raw : instance of RawYI
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: TODO
    """
    return RawYI(hdf5_filename, acquisition_name, preload=preload,
                  verbose=verbose)

def read_yi_acquisition_names(file_name):
    with h5py.File(file_name, 'r') as hdf5_file:
        acq_names = list(hdf5_file[YI.PATH_ACQUISITIONS].keys())
    return acq_names

@fill_doc
class RawYI(BaseRaw):
    """Raw object from YI hdf5 file.

    Parameters
    ----------
    hdf5_filenname : str
        Path to the hdf5 data
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    
    @verbose
    def __init__(self, hdf5_filename, acquisition_name='default',
                 preload=False, verbose=None):
        self.acquisition_name = acquisition_name
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            try:
                acquisition = hdf5_file[path.join(
                    YI.PATH_ACQUISITIONS,
                    acquisition_name
                )]
            except KeyError:
                raise KeyError(
                    'Could not find acquisition "{}" in file "{}"'.format(
                        acquisition_name,
                        hdf5_filename,
                    )
                )
            info = _compose_meas_info(
                hdf5_file,
                acquisition=acquisition,
            )
            last_samps = [acquisition['data'].shape[0] - 1]
        first_samps = [0]
        raw_extras = list()
        sample_info = dict(n_samp=last_samps, n_chan=info['nchan'])
        raw_extras.append(sample_info)

        super(RawYI,self).__init__(
            info,
            preload,
            first_samps=first_samps,
            last_samps=last_samps,
            filenames=(hdf5_filename,),
            raw_extras=raw_extras,
            orig_format='double',
            buffer_size_sec=300,
            verbose=verbose
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        with h5py.File(self._filenames[fi], 'r') as hdf5_file:
            acquisition = hdf5_file[path.join(
                YI.PATH_ACQUISITIONS,
                'default',#TODO make this adjustable
            )]
            # Row = time sample
            acq_data = acquisition['data'][start:stop, idx].T

            if mult is not None:
                np.dot(mult, acq_data, data)
            else:
                np.multiply(cals, acq_data, data)
