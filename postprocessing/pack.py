# include modules to the path
import sys, os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'session'))

import os, json, h5py, time
import numpy as np
import scipy.ndimage as ndi
from scipy import signal
from head_direction import head_direction
from spatial import place_field_2D, map_stats, get_field_patches, best_match_rotation_polar
from spatial import bins2meters, cart2pol, pol2cart
from spiketrain import instantaneous_rate, spike_idxs
from spiking_metrics import mean_firing_rate, isi_cv, isi_fano
from session.utils import get_sessions_list, get_sampling_rate, cleaned_epochs
from session.adapters import load_clu_res, H5NAMES, create_dataset


def pack(session_path):
    """
    Pack independent tracking datasets into a single HDF5 file.
    
    File has the following structure:
    
    /raw
        /positions      - raw positions from .csv
        /events         - raw events from .csv
        /sounds         - raw sounds from .csv
        /islands        - raw island infos from .csv (if exists)
    /processed
        /timeline       - matrix of [time, x, y, speed, HD, trial_no, sound_id] sampled at 100Hz,
                          data is smoothed using gaussian kernels,
                          inter-trial intervals have trial_no = 0
        /trial_idxs     - matrix of trial indices to timeline
        /sound_idxs     - matrix of sound indices to timeline
        
    each dataset has an attribute 'headers' with the description of columns.
    """
    params_file = [x for x in os.listdir(session_path) if x.endswith('.json')][0]

    with open(os.path.join(session_path, params_file)) as json_file:
        parameters = json.load(json_file)
    
    h5name = os.path.join(session_path, '%s.h5' % params_file.split('.')[0])
    with h5py.File(h5name, 'w') as f:  # overwrite mode

        # -------- save raw data ------------
        raw = f.create_group('raw')
        raw.attrs['parameters'] = json.dumps(parameters)

        for ds_name in ['positions', 'events', 'sounds', 'islands']:
            filename = os.path.join(session_path, '%s.csv' % ds_name)
            if not os.path.exists(filename):
                continue
                
            with open(filename) as ff:
                headers = ff.readline()
            data = np.loadtxt(filename, delimiter=',', skiprows=1)

            ds = raw.create_dataset(ds_name, data=data)
            ds.attrs['headers'] = headers
        
        # TODO - saving contours! and get file names from the config
#         with open(os.path.join(session_path, '%s.csv' % 'contours')) as ff:
#             data = ff.readlines()
        
#         headers = data[0]   # skip headers line
#         contours = [[(x.split(':')[0], x.split(':')[1]) for x in contour.split(',')] for contour in data[1:]]
#         contours = [np.array(contour) for contour in contours]

        # read raw data and normalize to session start
        positions = np.array(f['raw']['positions'])
        s_start, s_end = positions[:, 0][0], positions[:, 0][-1]
        positions[:, 0] = positions[:, 0] - s_start
        events = np.array(f['raw']['events'])
        events[:, 0] = events[:, 0] - s_start
        sounds = np.array(f['raw']['sounds'])
        sounds[:, 0] = sounds[:, 0] - s_start

        # squeeze - if session was interrupted, adjust times
        # to have a continuous timeline
        end_idxs = np.where(events[:, 5] == -1)[0]
        if len(end_idxs) > 1:
            # diffs in time beetween pauses
            deltas = [events[idx + 1][0] - events[idx][0] for idx in end_idxs[:-1]]

            for df, delta in zip(end_idxs, deltas):  # squeezing events
                events[df+1:][:, 0] = events[df+1:][:, 0] - delta

            end_idxs = np.where(np.diff(sounds[:, 0]) > 20)[0]  # squeezing sounds
            for df, delta in zip(end_idxs, deltas):
                sounds[df+1:][:, 0] = sounds[df+1:][:, 0] - delta

            end_idxs = np.where(np.diff(positions[:, 0]) > 20)[0]  # squeezing positions - more than 20? secs pauses
            for df, delta in zip(end_idxs, deltas):
                positions[df+1:][:, 0] = positions[df+1:][:, 0] - delta
            parameters['experiment']['timepoints'] = [positions[df+1][0] for df in end_idxs]  # update session parameters
            parameters['experiment']['session_duration'] = positions[-1][0]

        # -------- save processed ------------
        proc = f.create_group('processed')
        proc.attrs['parameters'] = json.dumps(parameters)

        # TODO remove outliers - position jumps over 20cm?
        #diffs_x = np.diff(positions[:, 1])
        #diffs_y = np.diff(positions[:, 2])
        #dists = np.sqrt(diffs_x**2 + diffs_y**2)
        #np.where(dists > 0.2 / pixel_size)[0]

        # convert timeline to 100 Hz
        time_freq = 100  # at 100Hz
        s_start, s_end = positions[:, 0][0], positions[:, 0][-1]
        times = np.linspace(s_start, s_end, int((s_end - s_start) * time_freq))
        pos_at_freq = np.zeros((len(times), 3))

        curr_idx = 0
        for i, t in enumerate(times):
            if curr_idx < len(positions) - 1 and \
                np.abs(t - positions[:, 0][curr_idx]) > np.abs(t - positions[:, 0][curr_idx + 1]):
                curr_idx += 1
            pos_at_freq[i] = (t, positions[curr_idx][1], positions[curr_idx][2])

        # save trials
        t_count = len(np.unique(events[events[:, -1] != 0][:, -2]))
        trials = np.zeros((t_count, 6))
        for i in range(t_count):
            t_start_idx = (np.abs(pos_at_freq[:, 0] - events[2*i][0])).argmin()
            t_end_idx = (np.abs(pos_at_freq[:, 0] - events[2*i + 1][0])).argmin()
            state = 0 if events[2*i + 1][-1] > 1 else 1

            trials[i] = (t_start_idx, t_end_idx, events[2*i][1], events[2*i][2], events[2*i][3], state)

        trial_idxs = proc.create_dataset('trial_idxs', data=trials)
        trial_idxs.attrs['headers'] = 't_start_idx, t_end_idx, target_x, target_y, target_r, fail_or_success'

        # save sounds
        sound_idxs = np.zeros((len(sounds), 2))
        left_idx = 0
        delta = 10**5
        for i in range(len(sounds)):
            while left_idx < len(pos_at_freq) and \
                    np.abs(sounds[i][0] - pos_at_freq[:, 0][left_idx]) < delta:
                delta = np.abs(sounds[i][0] - pos_at_freq[:, 0][left_idx])
                left_idx += 1

            sound_idxs[i] = (left_idx, sounds[i][1])
            delta = 10**5

        sound_idxs = proc.create_dataset('sound_idxs', data=sound_idxs)
        sound_idxs.attrs['headers'] = 'timeline_idx, sound_id'

        # building timeline
        width = 50  # 100 points ~= 1 sec with at 100Hz
        kernel = signal.gaussian(width, std=(width) / 7.2)

        x_smooth = np.convolve(pos_at_freq[:, 1], kernel, 'same') / kernel.sum()
        y_smooth = np.convolve(pos_at_freq[:, 2], kernel, 'same') / kernel.sum()

        # speed
        dx = np.sqrt(np.square(np.diff(x_smooth)) + np.square(np.diff(y_smooth)))
        dt = np.diff(pos_at_freq[:, 0])
        speed = np.concatenate([dx/dt, [dx[-1]/dt[-1]]])

        # head direction
        temp_tl = np.column_stack([pos_at_freq[:, 0], x_smooth, y_smooth, speed])
        hd = head_direction(temp_tl)

        # trial numbers
        trials_data = np.zeros(len(temp_tl))

        for i, trial in enumerate(trials):
            idx1, idx2 = trial[0], trial[1]
            trials_data[int(idx1):int(idx2)] = i + 1

        # sounds played
        sound_tl = np.zeros(len(temp_tl))
        curr_sound_idx = 0
        for i in range(len(temp_tl)):
            if curr_sound_idx + 1 >= len(sounds):
                break

            if temp_tl[i][0] > sounds[curr_sound_idx][0]:
                curr_sound_idx += 1
            sound_tl[i] = sounds[curr_sound_idx][1]

        timeline = proc.create_dataset('timeline', data=np.column_stack(\
                     [pos_at_freq[:, 0], x_smooth, y_smooth, speed, hd, trials_data, sound_tl]
                   ))
        timeline.attrs['headers'] = 'time, x, y, speed, hd, trial_no, sound_ids'
        
    return h5name


def write_units(sessionpath):
    filebase = os.path.basename(os.path.normpath(sessionpath))
    h5name  = os.path.join(sessionpath, filebase + '.h5')

    # loading unit data
    units = load_clu_res(sessionpath)  # spikes are in samples, not seconds
    sampling_rate = get_sampling_rate(sessionpath)
    
    # loading trajectory
    with h5py.File(h5name, 'r') as f:
        tl = np.array(f['processed']['timeline'])  # time, X, Y, speed, HD, trials, sounds
        
    # packing
    with h5py.File(h5name, 'a') as f:
        if not 'units' in f:
            f.create_group('units')

    for electrode_idx in units.keys():
        unit_idxs = units[electrode_idx]

        for unit_idx, spiketrain in unit_idxs.items():
            unit_name = '%s-%s' % (electrode_idx, unit_idx)

            s_times = spiketrain/sampling_rate
            i_rate = instantaneous_rate(s_times, tl[:, 0])
            s_idxs = spike_idxs(s_times, tl[:, 0])

            with h5py.File(h5name, 'a') as f:
                if not unit_name in f['units']:
                    grp = f['units'].create_group(unit_name)

            create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.spike_times, s_times)
            create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.inst_rate, i_rate)
            create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.spike_idxs, s_idxs)


def write_spiking_metrics(sessionpath):
    filebase = os.path.basename(os.path.normpath(sessionpath))
    h5name  = os.path.join(sessionpath, filebase + '.h5')
    epochs = cleaned_epochs(sessionpath)
    
    with h5py.File(h5name, 'r') as f:
        unit_names = [name for name in f['units']]
        
    for unit_name in unit_names:
        with h5py.File(h5name, 'r') as f:
            st = np.array(f['units'][unit_name][H5NAMES.spike_times['name']])
            
        mfr_vals    = np.zeros(len(epochs))
        isi_cv_vals = np.zeros(len(epochs))
        isi_fn_vals = np.zeros(len(epochs))
        for i, epoch in enumerate(epochs):
            st_cut = st[(st > epoch[0]) & (st < epoch[1])]
            mfr_vals[i] = mean_firing_rate(st_cut)
            isi_cv_vals[i] = isi_cv(st_cut)
            isi_fn_vals[i] = isi_fano(st_cut)
            
        create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.mfr, np.array(mfr_vals))
        create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.isi_cv, np.array(isi_cv_vals))
        create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.isi_fano, np.array(isi_fn_vals))


def write_spatial_metrics(sessionpath):
    metric_names = (H5NAMES.o_maps, H5NAMES.f_maps, H5NAMES.sparsity, H5NAMES.selectivity, \
                    H5NAMES.spat_info, H5NAMES.peak_FR, H5NAMES.f_patches, H5NAMES.f_COM, \
                    H5NAMES.pfr_center, H5NAMES.occ_info, H5NAMES.o_patches, H5NAMES.o_COM)
    xy_range = [-0.5, 0.5, -0.5, 0.5]  # make fixed for cross-comparisons
    bin_size = 0.02
    filebase = os.path.basename(os.path.normpath(sessionpath))
    h5name  = os.path.join(sessionpath, filebase + '.h5')
    epochs = cleaned_epochs(sessionpath)

    with h5py.File(h5name, 'r') as f:
        tl = np.array(f['processed']['timeline'])  # time, X, Y, speed, etc.
        unit_names = [name for name in f['units']]
        
    s_rate_pos = round(1.0 / np.diff(tl[:, 0]).mean())
    run_idxs = np.where(tl[:, 3] > 0.04)[0]

    for unit_name in unit_names:
        with h5py.File(h5name, 'r') as f:
            spk_idxs = np.array(f['units'][unit_name][H5NAMES.spike_idxs['name']])
        spk_idxs = np.intersect1d(spk_idxs, run_idxs)

        collected = []
        for epoch in epochs:
            epoch_idxs = np.where((tl[:, 0] > epoch[0]) & (tl[:, 0] < epoch[1]))[0]

            # filter for epoch and speed > 4cm/s
            unit_pos = tl[np.intersect1d(spk_idxs, epoch_idxs)][:, 1:3]
            traj_pos = tl[epoch_idxs][:, 1:3]

            # compute 2D maps: occupancy and firing rate (place fields)
            #xy_range = [tl[:, 1].min(), tl[:, 1].max(), tl[:, 2].min(), tl[:, 2].max()]
            o_map, s1_map, s2_map, f_map = place_field_2D(traj_pos, unit_pos, s_rate_pos, bin_size=bin_size, xy_range=xy_range)

            # firing map metrics
            sparsity, selectivity, spat_info, peak_FR = map_stats(f_map, o_map)

            # place field metrics
            patches = get_field_patches(f_map)  # 2D matrix, patches labeled according to the size
            #f_sizes = np.bincount(patches.flat)[1:]  # 1D array of field sizes, sorted
            if f_map.max() == 0:
                f_COM_rho, f_COM_phi, pfr_rho, pfr_phi = 0, 0, 0, 0
            else:
                x_in_b, y_in_b = ndi.center_of_mass(f_map, labels=patches, index=1)  # largest field COM, in bins
                f_COM_rho, f_COM_phi = cart2pol(*bins2meters(x_in_b, y_in_b, xy_range))  # largest field COM, in polar coords.
                x, y = np.where(f_map == np.max(f_map))  # location of the peak unit firing, in bins
                pfr_rho, pfr_phi = cart2pol(*bins2meters(x[0], y[0], xy_range))  # location of the peak unit firing, in polar
            
            # same for occupancy
            _, _, occ_info, _ = map_stats(o_map, o_map)
            o_patches = get_field_patches(o_map)  # 2D matrix, patches labeled according to the size
            x, y = ndi.center_of_mass(o_map, labels=o_patches, index=1)  # largest field COM, in bins
            o_COM_rho, o_COM_phi = cart2pol(*bins2meters(x, y, xy_range))     # largest field COM, in polar coords.

            # order should match metric_names defined above
            collected.append([o_map, f_map, sparsity, selectivity, spat_info, peak_FR, \
                patches, (f_COM_rho, f_COM_phi), (pfr_rho, pfr_phi), occ_info, \
                o_patches, (o_COM_rho, o_COM_phi)])
            
        for i in range(len(collected[0])):  # iterate over metrics
            dataset = np.array([x[i] for x in collected])  # metric data for each epoch
            create_dataset(h5name, '/units/%s' % unit_name, metric_names[i], dataset)


def write_best_match_rotation(sessionpath):
    filebase = os.path.basename(os.path.normpath(sessionpath))
    h5name  = os.path.join(sessionpath, filebase + '.h5')

    with h5py.File(h5name, 'r') as f:
        unit_names = [name for name in f['units']]

    for unit_name in unit_names:

        # assuming maps for first 3 epochs are already there
        with h5py.File(h5name, 'r') as f:
            maps = np.array(f['units'][unit_name][H5NAMES.f_maps['name']])

        corr_profiles = np.zeros((3, 360))
        for i, idxs in enumerate([(0, 1), (1, 2), (0, 2)]):
            corr_profiles[i], phi = best_match_rotation_polar(maps[idxs[0]], maps[idxs[1]])

        create_dataset(h5name, '/units/%s' % unit_name, H5NAMES.best_m_rot, corr_profiles)