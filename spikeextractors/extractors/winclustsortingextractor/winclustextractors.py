from spikeextractors import SortingExtractor
import numpy as np
import glob

file_path = 'E:/Rat883/200317_Rat883-16/Neuralynx/TT11'


class WinClustSortingExtractor(SortingExtractor):
    extractor_name = 'WinClustSortingExtractor'
    installed = True
    is_writable = True

    def __init__(self, file_path, sampling_frequency=None):
        SortingExtractor.__init__(self)
        self.file_path = file_path
        self.cl_files = glob.glob(self.file_path + "/cl-maze*.*", recursive=True)

        self._features = {
            'MaxHeight': [],
            'MaxWidth': [],
            'XPos': [],
            'YPos': [],
            'Timestamp': [],
            'SpikeID': [],
            'X': {'Peak': [], 'PreValley': [], 'Energy': []},
            'Y': {'Peak': [], 'PreValley': [], 'Energy': []},
            'A': {'Peak': [], 'PreValley': [], 'Energy': []},
            'B': {'Peak': [], 'PreValley': [], 'Energy': []}
        }

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
                for index, subfeature in enumerate(self._features[feature]):
                    num = _id_code_from_id(feature) + 1
                    self._features[feature][subfeature] = self.sorted_events_time[:, (num + (index * 5))]

    def get_unit_ids(self):
        letter_ids = 'XYAB'
        return list(letter_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        return self.sorted_events_time[:, -1]


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
