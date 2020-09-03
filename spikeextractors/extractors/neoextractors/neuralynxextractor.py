from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor
from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args

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
    The neuralynx extractor is wrapped from neo.rawio.NeuralynxRawIO.
    
    Parameters
    ----------
    dirname: str
        The neuralynx folder that contain all neuralynx files ('nse', 'ncs', 'nev', 'ntt')
    block_index: None or int
        If the underlying dataset has several blocks the index must be specified.
    seg_index: None or int
        If the underlying dataset has several segments the index must be specified.
    
    """

    extractor_name = 'NeuralynxRecording'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'


class NeuralynxNrdRecordingExtractor(RecordingExtractor):
    """
    Extractor to read from Neuralynx Raw Data files (*.nrd). These are the dump of
    the raw, unfiltered A/D records into one huge file per recording session.

    The main nrd parsing code has been adapted from the neurapy collection by Kaushik Ghose
    https://github.com/kghose/neurapy

    TODO:
        (Optional) Read the Neuralynx config file to determine electrode configuration.
    """
    extractor_name = 'NeuralynxNrdRecording'
    installed = True  # check at class level if installed or not
    is_writable = False

    def __init__(self, file_name, error_checking=False):
        """
        Parameters
        ----------
        file_name: str
            The *.nrd file name (with path)
        error_checking: bool
            Whether or not to implement CRC error checking for each record.
            Error checking is slower, but returns accurate records.
        """
        assert Path(file_name).suffix == '.nrd', "{} is not a Neuralynx nrd file (.nrd)".format(file_name)

        RecordingExtractor.__init__(self)
        self._nrd_file_name = file_name
        self._error_checking = error_checking
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

    def seek_packet(self):
        """Skip forward until we find the STX magic number."""
        # Read in 32bit increments until the magic number is found
        f = self._nrd_fid
        start = f.tell()
        pkt = f.read(4)
        while len(pkt) == 4:
            if pkt == b'\x00\x08\x00\x00':  # Magic number 2048 0x0800
                f.seek(-4, 1)  # Realign
                break
            pkt = f.read(4)
        stop = f.tell()
        return stop - start

    @check_get_traces_args
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

        if start_frame is None:
            start_frame = 0

        if end_frame is None:
            end_frame = self._approx_num_frames

        if (start_frame + (end_frame - start_frame)) > self._approx_num_frames:
            end_frame = self._approx_num_frames

        num_frames_to_read = end_frame - start_frame

        self._nrd_fid.seek(16*1024 + start_frame * self._record_size)   # Skip to start frame

        if not self._error_checking:
            # Read in 32bit increments until the magic number is found
            pkt = self._nrd_fid.read(4)
            while len(pkt) == 4:
                if pkt == b'\x00\x08\x00\x00':  # Magic number 2048 0x0800
                    self._nrd_fid.seek(-4, 1)  # Realign
                    break
                pkt = self._nrd_fid.read(4)

            these_packets = np.fromfile(self._nrd_fid, dtype=self._nrd_packet, count=num_frames_to_read)
            return np.transpose(these_packets['data'][:, channel_ids])
        else:
            pkt_cnt = 0
            garbage_bytes = 0
            stx_err_cnt = 0
            pkt_id_err_cnt = 0
            pkt_size_err_cnt = 0
            pkt_ts_err_cnt = 0
            pkt_crc_err_cnt = 0

            buffer_size = 100000
            if buffer_size > num_frames_to_read:
                buffer_size = num_frames_to_read

            last_ts = 0

            garbage_bytes += self.seek_packet()

            return_array = np.empty((1, len(channel_ids)))
            return_array.fill(np.nan)
            these_packets = np.fromfile(self._nrd_fid, dtype=self._nrd_packet, count=buffer_size)
            while these_packets.size > 0:
                all_packets_good = True
                packets_read = these_packets.size

                idx = np.argwhere(these_packets['stx'] != 2048)
                if idx.size > 0:
                    stx_err_cnt += 1
                    all_packets_good = False
                    max_good_packets = idx[0]
                    these_packets = these_packets[:max_good_packets]

                if these_packets.size > 0:
                    idx = np.argwhere(these_packets['pkt_id'] != 1)
                    if idx.size > 0:
                        pkt_id_err_cnt += 1
                        all_packets_good = False
                        max_good_packets = idx[0]
                        these_packets = these_packets[:max_good_packets]

                if these_packets.size > 0:
                    idx = np.argwhere(these_packets['pkt_data_size'] != 10 + self._num_channels)
                    if idx.size > 0:
                        pkt_size_err_cnt += 1
                        all_packets_good = False
                        max_good_packets = idx[0]
                        these_packets = these_packets[:max_good_packets]

                if these_packets.size > 0:
                    # crc computation
                    field32 = np.vstack([these_packets[k].T for k in self._nrd_packet.fields.keys()]).astype('I')
                    crc = np.zeros(these_packets.size, dtype='I')
                    for idx in range(field32.shape[0]):
                        crc ^= field32[idx, :]
                    idx = np.argwhere(crc != 0)
                    if idx.size > 0:
                        pkt_crc_err_cnt += 1
                        all_packets_good = False
                        max_good_packets = idx[0]
                        these_packets = these_packets[:max_good_packets]

                if these_packets.size > 0:
                    ts = (these_packets['timestamp high'].astype('uint64') << 32) | these_packets['timestamp low']
                    bad_idx = -1
                    if last_ts > ts[0]:  # Time stamps out of order at buffer boundary
                        bad_idx = 0
                    else:
                        idx = np.argwhere(ts[:-1] > ts[1:])
                        if idx.size > 0:
                            bad_idx = idx[0] + 1
                    if bad_idx > -1:
                        warnings.warn('Out of order timestamp {:d}'.format(int(ts[bad_idx])), UserWarning)
                        pkt_ts_err_cnt += 1
                        all_packets_good = False
                        max_good_packets = bad_idx
                        these_packets = these_packets[:max_good_packets]
                        ts = ts[:max_good_packets]

                if these_packets.size > 0:
                    last_ts = ts[-1]  # Ready for the next read

                if not all_packets_good:
                    self._nrd_fid.seek((these_packets.size - packets_read) * self._record_size + 4, 1)  # Rewind all the way except 32 bits
                    garbage_bytes += self.seek_packet()

                if np.all(np.isnan(return_array)):
                    return_array = these_packets['data'][:, channel_ids]
                else:
                    return_array = np.concatenate((return_array, these_packets['data'][:, channel_ids]))

                pkt_cnt += these_packets.size
                if pkt_cnt >= num_frames_to_read:  # NOTE: This may give us upto buffer_size -1 more packets than we want.
                    break

                these_packets = np.fromfile(self._nrd_fid, dtype=self._nrd_packet, count=buffer_size)

            if np.shape(return_array)[0] > num_frames_to_read:
                return_array = return_array[range(num_frames_to_read), :]

            return np.transpose(return_array)



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
