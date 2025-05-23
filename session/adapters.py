import numpy as np
import os
import h5py
from scipy import signal
from scipy.signal import butter, sosfilt


COLORS  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:gray'] 
EPOCH_NAMES = ('Original', 'Conflict', 'Control', 'All')

class H5NAMES:
    inst_rate =   {'name': 'inst_rate', 'dims': ['instantaneous firing rate at ~100Hz']}
    spike_times = {'name': 'spike_times', 'dims': ['spike times in seconds']}
    spike_idxs =  {'name': 'spike_idxs', 'dims': ['indices to timeline when spikes occured']}
    mfr =         {'name': 'mean_firing_rate', 'dims': ['epochs: original, conflict, control and all']}
    isi_cv =      {'name': 'isi_coeff_var', 'dims': ['epochs: original, conflict, control and all']}
    isi_fano =    {'name': 'isi_fano_factor', 'dims': ['epochs: original, conflict, control and all']}
    o_maps =      {'name': 'occupancy_maps', 'dims': [
        'epochs: original, conflict, control and all', 'X, bins', 'Y, bins'
    ]}
    f_maps =      {'name': 'firing_rate_maps', 'dims': [
        'epochs: original, conflict, control and all', 'X, bins', 'Y, bins'
    ]}
    sparsity =    {'name': 'sparsity', 'dims': ['epochs: original, conflict, control and all']}
    selectivity = {'name': 'selectivity', 'dims': ['epochs: original, conflict, control and all']}
    spat_info =   {'name': 'spatial_information', 'dims': ['epochs: original, conflict, control and all']}
    peak_FR =     {'name': 'peak_firing_rate', 'dims': ['epochs: original, conflict, control and all']}
    f_patches =   {'name': 'field_patches', 'dims': [
        'epochs: original, conflict, control and all', 'X, bins', 'Y, bins'
    ]}
    f_sizes =     {'name': 'field_sizes', 'dims': ['epochs: original, conflict, control and all']}
    f_COM =       {'name': 'field_center_of_mass', 'dims': ['epochs: original, conflict, control and all', 'rho, phi in polar coords.']}
    pfr_center =  {'name': 'field_center_of_firing', 'dims': ['epochs: original, conflict, control and all', 'rho, phi in polar coords.']}
    occ_info =    {'name': 'occupancy_information', 'dims': ['epochs: original, conflict, control and all']}
    o_patches =   {'name': 'occupancy_patches', 'dims': [
        'epochs: original, conflict, control and all', 'X, bins', 'Y, bins'
    ]}
    o_COM =       {'name': 'occupancy_center_of_mass', 'dims': ['epochs: original, conflict, control and all', 'rho, phi in polar coords.']}
    best_m_rot  = {'name': 'best_match_rotation', 'dims': ['match between: A-B, B-C, A-C', 'correlation profile']}


def load_clu_res(where):
    """
    Neurosuite files:
    
    dat     - raw signal in binary (usually int16) format as a matrix channels x signal
    lfp     - raw signal, downsampled (historically to 1250Hz)
    fet     - list of feature vectors for every spike for a particular electrode
    spk     - list of spike waveforms for every spike for a particular electrode, binary
    res     - spike times in samples for all clusters (units) from a particular electrode
    clu     - list of cluster (unit) numbers for each spike from 'res'

    Load spike times from 'clu' (clusters) and 'res' (spike times) files generated by KlustaKwik.

    :param where:       path to the folder
    :param filebase:    base name of the file (like 'foo' in 'foo.clu.3')
    :param index:       index of the file (like '3' in 'foo.clu.3')
    :return:            a dict in a form like {<clustered_unit_no>: <spike_times>, ...}
    """
    filebase = os.path.basename(where)
    clu_files = [f for f in os.listdir(where) if f.find('.clu.') > 0]
    if not len(clu_files) > 0:
        return {}
    
    idxs = [int(x.split('.')[2]) for x in clu_files]  # electrode indexes
    
    all_units = {}
    for idx in idxs:
        clu_file = os.path.join(where, '.'.join([filebase, 'clu', str(idx)]))
        res_file = os.path.join(where, '.'.join([filebase, 'res', str(idx)]))

        if not os.path.isfile(clu_file) or not os.path.isfile(res_file):
            continue

        cluster_map = np.loadtxt(clu_file, dtype=np.uint16)  # uint16 for clusters
        all_spikes = np.loadtxt(res_file, dtype=np.uint64)   # uint64 for spike times

        cluster_map = cluster_map[1:]  # remove the first element - number of clusters

        result = {}
        for cluster_no in np.unique(cluster_map)[1:]:  # already sorted / remove 1st cluster - noise
            result[cluster_no] = all_spikes[cluster_map == cluster_no]
            
        all_units[idx] = result

    return all_units


def create_dataset(h5name, where, descriptor, dataset):
    """
    h5name       path to an HDF5 file
    where        path inside the file
    descriptor   H5NAMES style descriptor of the dataset
    dataset      numpy array to store
    """
    with h5py.File(h5name, 'a') as f:
        target_group = f[where]

        if descriptor['name'] in target_group:  # overwrite mode
            del target_group[descriptor['name']]
            
        ds = target_group.create_dataset(descriptor['name'], data=dataset)
        for i, dim in enumerate(descriptor['dims']):
            ds.attrs['dim%s' % i] = dim
            
            
class DatProcessor:
    
    def __init__(self, dat_file, channel_count=64):
        # read this from XML file
        self.s_rate = 30000
        self.ch_no = channel_count
        self.dat_file = dat_file
        
    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            sos = butter(order, [low, high], analog=False, btype='band', output='sos')
            return sos        

        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y
    
    def read_block_from_dat(self, duration, offset):
        """
        duration      in seconds
        offset        in seconds
        """
        count = self.s_rate * self.ch_no * duration  # number of values to read
        offset_in_bytes = offset * self.s_rate * self.ch_no * 2  # assuming int16 is 2 bytes
        block = np.fromfile(self.dat_file, dtype=np.int16, count=int(count), offset=int(offset_in_bytes))
        return block.reshape([int(self.s_rate * duration), self.ch_no])
    
    def get_single_channel(self, channel_no, block_duration=1):  # block duration 1 sec
        size = os.path.getsize(self.dat_file)
        samples_no = size / (64 * 2)

        raw_signal = np.zeros(int(samples_no))  # length in time: samples_no / sample_rate
        offset = 0

        while offset < samples_no / self.s_rate - block_duration:
            block = self.read_block_from_dat(block_duration, offset)  # read in 1 sec blocks
            raw_signal[self.s_rate*offset:self.s_rate*(offset + block_duration)] = block[:, channel_no]
            offset += block_duration

        return raw_signal