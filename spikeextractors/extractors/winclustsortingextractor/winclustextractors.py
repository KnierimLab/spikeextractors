from spikeextractors import SortingExtractor
import numpy as np
import glob
from spikeextractors.extraction_tools import cast_start_end_frame

file_path = 'E:/Rat883/200317_Rat883-16/Neuralynx/TT11'


class WinClustSortingExtractor(SortingExtractor):
    extractor_name = 'WinClustSortingExtractor'
    installed = True
    is_writable = True

    def __init__(self, dir_path, sampling_frequency=10000000):
        SortingExtractor.__init__(self)
        self.dir_path = dir_path
        self.cl_files = glob.glob(self.file_path + "/cl-maze*.*", recursive=True)
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
        self.cluster_starts = np.cumsum([len(y) for y in self.struct])
        self.all_events = np.vstack(self.struct)
        self.sorted_events_time = self.all_events[np.argsort(self.all_events[:, -1])]

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
        converted_train = (self.sorted_events_time[:, -1] - self.start_time) * self._sampling_frequency
        ind = np.where((converted_train >= start_frame) & (converted_train <= end_frame))
        return np.rint(converted_train[ind]).astype(int)


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
