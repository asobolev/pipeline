import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..', '..'))

from imports import *
from target import build_tgt_matrix
import pandas as pd


def load_session_data(session, load_units=True, load_aeps=True, load_moseq=True):
    all_areas = ['A1', 'PPC', 'HPC']

    animal      = session.split('_')[0]
    sessionpath = os.path.join(source, animal, session)
    h5_file     = os.path.join(sessionpath, session + '.h5')
    aeps_file   = os.path.join(sessionpath, 'AEPs.h5')
    moseq_file  = os.path.join(sessionpath, 'moseq.h5')
    report_path = os.path.join(report, 'PSTH', session)
    if not os.path.exists(report_path):
        os.makedirs(report_path)
        
    # load timeline and configuration
    with h5py.File(h5_file, 'r') as f:
        tl = np.array(f['processed']['timeline'])  # time, X, Y, speed, etc.
        trials = np.array(f['processed']['trial_idxs'])  # t_start_idx, t_end_idx, x_tgt, y_tgt, r_tgt, result
        cfg = json.loads(f['processed'].attrs['parameters'])

    # load units
    unit_names, single_units, spike_times = [], {}, {}
    if load_units:
        with h5py.File(h5_file, 'r') as f:
            unit_names = [x for x in f['units']]

        with h5py.File(h5_file, 'r') as f:
            for unit_name in unit_names:
                spike_times[unit_name] = np.array(f['units'][unit_name][H5NAMES.spike_times['name']])
                single_units[unit_name] = np.array(f['units'][unit_name][H5NAMES.inst_rate['name']])
                #single_units[unit_name] = instantaneous_rate(unit_times, tl[:, 0], k_width=50)
        
    # load AEPs
    areas, aeps, aeps_events, lfp = [], {}, [], {}
    AEP_metrics_lims, AEP_metrics_raw, AEP_metrics_norm = {}, {}, {}
    tgt_matrix = []
    
    if load_aeps:
        with h5py.File(aeps_file, 'r') as f:
            for area in all_areas:
                if not area in f:
                    continue
                aeps[area] = np.array(f[area]['aeps'])
            aeps_events = np.array(f['aeps_events'])

        areas = list(aeps.keys())
        # TODO find better way. Remove outliers
        if 'A1' in areas:
            aeps['A1'][aeps['A1'] >  5000]   =   5000
            aeps['A1'][aeps['A1'] < -5000]   =  -5000
        if 'PPC' in areas:
            aeps['PPC'][aeps['PPC'] >  1500] =   1500
            aeps['PPC'][aeps['PPC'] < -1500] =  -1500
        if 'HPC' in areas:
            aeps['HPC'][aeps['HPC'] >  1500] =   1500
            aeps['HPC'][aeps['HPC'] < -1500] =  -1500
        aeps[areas[0]].shape

        # load LFP
        lfp = {}
        with h5py.File(aeps_file, 'r') as f:
            for area in areas:
                if 'LFP' in f[area]:
                    lfp[area] = np.array(f[area]['LFP'])

        # load AEP metrics
        AEP_metrics_lims = dict([(area, {}) for area in areas])
        AEP_metrics_raw  = dict([(area, {}) for area in areas])
        AEP_metrics_norm = dict([(area, {}) for area in areas])
        with h5py.File(aeps_file, 'r') as f:
            for area in areas:
                grp = f[area]

                for metric_name in grp['raw']:
                    AEP_metrics_raw[area][metric_name]  = np.array(grp['raw'][metric_name])
                    AEP_metrics_norm[area][metric_name] = np.array(grp['norm'][metric_name])
                    AEP_metrics_lims[area][metric_name] = [int(x) for x in grp['raw'][metric_name].attrs['limits'].split(',')]

        # build target matrix
        tgt_matrix = build_tgt_matrix(tl, trials, aeps_events)

    # load moseq
    moseq = []
    if load_moseq:
        with h5py.File(moseq_file, 'r') as f:
            moseq_matrix  = np.array(f['moseq'])
            moseq_headers = f['moseq'].attrs['headers']
        moseq_headers = moseq_headers.split(',')
        moseq_headers = [moseq_headers[0]] + [x[1:] for x in moseq_headers[1:]]
        moseq = pd.DataFrame(moseq_matrix, columns=moseq_headers)
            
    return {
        'tl': tl,
        'trials': trials,
        'cfg': cfg,
        'areas': areas,
        'aeps': aeps,
        'aeps_events': aeps_events,
        'lfp': lfp,
        'AEP_metrics_lims': AEP_metrics_lims,
        'AEP_metrics_raw': AEP_metrics_raw,
        'AEP_metrics_norm': AEP_metrics_norm,
        'tgt_matrix': tgt_matrix,
        'single_units': single_units,
        'spike_times': spike_times,
        'unit_names': unit_names,
        'animal': animal,
        'aeps_file': aeps_file,
        'moseq_file': moseq_file,
        'h5_file': h5_file,
        'report_path': report_path,
        'moseq': moseq
    }
