from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor
from spikeextractors import RecordingExtractor

import numpy as np
import math
from pathlib import Path
import warnings

try:
    import neo

    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    """
    The neruralynx extractor is wrapped from neo.rawio.NeuralynxRawIO.
    
    Parameters
    ----------
    dirname: str
        The neuralynx folder that contain all neuralynx files ('nse', 'ncs', 'nev', 'ntt')
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.
    
    """

    extractor_name = 'NeuralynxRecording'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'


class NeuralynxNrdRecordingExtractor(RecordingExtractor):
    extractor_name = 'NeuralynxNrdRecording'
    installed = True  # check at class level if installed or not
    is_writable = False

    def __init__(self, file_name):
        RecordingExtractor.__init__(self)
        self._nrd_file_name = file_name
        self._nrd_fid = open(file_name, 'rb')
        self._file_size = Path(file_name).stat().st_size

        # Read 16 KB header
        self._nrd_hdr = self._nrd_fid.read(16 * 1024).decode().strip('\00')

        # Find parameters from header
        hdr = self._nrd_hdr.split()
        self._file_version = hdr[hdr.index('-FileVersion') + 1]
        self._date_created = hdr[hdr.index('-TimeCreated') + 1]
        self._time_created = hdr[hdr.index('-TimeCreated') + 2]
        self._date_closed = hdr[hdr.index('-TimeClosed') + 1]
        self._time_closed = hdr[hdr.index('-TimeClosed') + 2]
        self._record_size = int(hdr[hdr.index('-RecordSize') + 1])
        self._sampling_frequency = int(hdr[hdr.index('-SamplingFrequency') + 1])
        self._num_channels = int(hdr[hdr.index('-NumADChannels') + 1])
        self._ad_max_value = int(hdr[hdr.index('-ADMaxValue') + 1])
        self._ad_bit_volts = float(hdr[hdr.index('-ADBitVolts') + 1])
        self._approx_num_frames = math.floor((self._file_size - (16*1024))/self._record_size)

        # Make NRD packet for reading
        self._nrd_packet = self.make_nrd_packet()

        pass

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        '''This function extracts and returns a trace from the recorded data from the
        given channels ids and the given start and end frame. It will return
        traces from within three ranges:

            [start_frame, start_frame+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_recording_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_recording_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Traces are returned in a 2D array that
        contains all of the traces from each channel with dimensions
        (num_channels x num_frames). In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        start_frame: int
            The starting frame of the trace to be returned (inclusive)
        end_frame: int
            The ending frame of the trace to be returned (exclusive)
        channel_ids: array_like
            A list or 1D array of channel ids (ints) from which each trace will be
            extracted

        Returns
        ----------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (num_channels x num_frames)
        '''

        pass

    def get_num_frames(self):
        '''This function returns the number of frames in the recording

        Returns
        -------
        num_frames: int
            Number of frames in the recording (duration of recording)
        '''

        warnings.warn("Number of frames returned assumes no errors during recording", UserWarning)
        return self._approx_num_frames

        pass

    def get_sampling_frequency(self):
        '''This function returns the sampling frequency in units of Hz.

        Returns
        -------
        fs: float
            Sampling frequency of the recordings in Hz
        '''

        return self._sampling_frequency

    def get_channel_ids(self):
        '''Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.

        Returns
        -------
        channel_ids: list
            Channel list

        '''
        return list(range(self._num_channels))

    # nrd packet format
    def make_nrd_packet(self):
        nrd_packet = np.dtype([
            ('stx', 'i'),
            ('pkt_id', 'i'),
            ('pkt_data_size', 'i'),
            ('timestamp high', 'I'),  # Neuralynx timestamp is ... in its own 32 bit world
            ('timestamp low', 'I'),
            ('status', 'i'),
            ('ttl', 'I'),
            ('extra', '10i'),
            ('data', '{:d}i'.format(self._num_channels)),
            ('crc', 'i')
        ])
        return nrd_packet


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'NeuralynxSorting'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'
