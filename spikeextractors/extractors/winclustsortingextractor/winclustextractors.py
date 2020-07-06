from spikeextractors import SortingExtractor
import numpy as np
import glob
from spikeextractors.extraction_tools import cast_start_end_frame

class WinClustSortingExtractor(SortingExtractor):
    extractor_name = 'WinClustSortingExtractor'
    installed = True
    is_writable = True

    def __init__(self, dir_path, sampling_frequency):  # sample frequency is in Hz while timestamps are in us.
        SortingExtractor.__init__(self)
        self.dir_path = dir_path
        self.cl_files = glob.glob(self.dir_path + "/cl-maze*.*", recursive=True)
        self._unit_ids = [0, 1, 2, 3]
        self._sampling_frequency = sampling_frequency
        self._features = {
            'MaxHeight': [],
            'MaxWidth': [],
            'XPos': [],
            'YPos': [],
            'Timestamp': [],
            'SpikeID': [],
            0: {'Peak': [], 'PreValley': [], 'Energy': []},
            1: {'Peak': [], 'PreValley': [], 'Energy': []},
            2: {'Peak': [], 'PreValley': [], 'Energy': []},
            3: {'Peak': [], 'PreValley': [], 'Energy': []}
        }
        with open(self.cl_files[0], 'r') as t1:
            self.start_time = int(t1.read().splitlines()[11])

        self.struct = []
        for CLFile in self.cl_files:
            f1 = open(CLFile, 'r')
            data = f1.read().splitlines()[13:]
            f1.close()
            for row in range(len(data)):
                data[row] = [float(x) for x in data[row].split(',')]
            self.struct.append(np.array(data))
        self.cluster_ends = np.cumsum([len(y) for y in self.struct]).tolist()
        self.cluster_ends.insert(0, 0)
        self.all_events = np.vstack(self.struct)
        self.sorted_events_time = self.all_events[np.argsort(self.all_events[:, -1])]
        self.converted_train = (self.sorted_events_time[:, -1] - self.start_time)/1.0e6 * self._sampling_frequency

        # Get neighborhood data (ie max peak per spike)
        self.peak_chan = np.argmax(self.sorted_events_time[:, 1:4], axis=1)

        # Populating the dictionary of spike features
        for feature in self._features:
            if isinstance(self._features[feature], list):
                res = [idx for idx, key in enumerate(self._features) if key == feature]
                self._features[feature] = self.sorted_events_time[:, (res[0] - 5)]
            elif isinstance(self._features[feature], dict):
                for index, sub_feature in enumerate(self._features[feature]):
                    num = feature + 1
                    self._features[feature][sub_feature] = self.sorted_events_time[:, (num + (index * 5))]

    def get_unit_ids(self):
        return self._unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        ind = np.where((self.converted_train >= start_frame) & (self.converted_train <= end_frame))
        return np.rint(self.converted_train[ind]).astype(int)

    def get_cluster_spike_train(self, cluster_number=None):
        start_point = self.cluster_ends[cluster_number - 1]
        end_point = self.cluster_ends[cluster_number] - 1
        cluster_train = (self.all_events[start_point:end_point, -1] - self.start_time) * self._sampling_frequency
        return cluster_train

    def get_neighbor_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        ind = np.where((self.peak_chan == unit_id) &
                       (self.sorted_events_time[:, -1] >= start_frame) &
                       (self.sorted_events_time[:, -1] <= end_frame)
                       )
        return np.rint(self.converted_train[ind]).astype(int)

# following functions are to interface between how we like to identify electrodes in a tt vs. how SI wants to ID units.


def _id_code_from_id(_id):
    if _id == 'X' or id == 'x':
        id_code = 0
    elif _id == 'Y' or id == 'y':
        id_code = 1
    elif _id == 'A' or id == 'a':
        id_code = 2
    elif _id == 'B' or id == 'b':
        id_code = 3
    else:
        id_code = None
    return id_code


def _id_from_id_code(id_code):
    if id_code == 0:
        _id = 'X'
    elif id_code == 1:
        _id = 'Y'
    elif id_code == 2:
        _id = 'A'
    elif id_code == 3:
        _id = 'B'
    else:
        _id = None
    return _id
