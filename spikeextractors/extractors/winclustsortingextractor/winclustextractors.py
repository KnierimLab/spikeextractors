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

        self.fields = {
            'SpikeID': [],
            'PeakX': [],
            'PeakY': [],
            'PeakA': [],
            'PeakB': [],
            'PreValleyX': [],
            'PreValleyY': [],
            'PreValleyA': [],
            'PreValleyB': [],
            'EnergyX': [],
            'EnergyY': [],
            'EnergyA': [],
            'EnergyB': [],
            'MaxHeight': [],
            'MaxWidth': [],
            'XPos': [],
            'YPos': [],
            'Timestamp': []
        }  # This is dictionary is currently unused...

        self.struct = []
        for CLFile in self.cl_files:
            f1 = open(CLFile, 'r')
            data = f1.read().splitlines()[13:]
            for row in range(len(data)):
                data[row] = [float(x) for x in data[row].split(',')]
            self.struct.append(np.array(data))
        self.cluster_starts = np.cumsum([len(y) for y in self.struct])
        self.all_events = np.vstack(self.struct)
        self.sorted_events_time = self.all_events[np.argsort(self.all_events[:, -1])]

    def get_unit_ids(self):
        letter_ids = 'XYAB'
        return list(letter_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        return self.sorted_events_time[:, -1]



