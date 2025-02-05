import os, json, h5py
import numpy as np
import xml.etree.ElementTree as ET

from datetime import datetime


def session_to_numbers(session_name):
    animal, e_type, e_date, e_time = session_name.split('_')

    dt = datetime.strptime('%s_%s' % (e_date, e_time), '%Y-%m-%d_%H-%M-%S')

    animal_code  = int(animal)
    session_code = (dt.year-2000)*10**6 + dt.month*10**4 + dt.day*10**2 + dt.hour
    return animal_code, session_code


def guess_filebase(sessionpath):
    return os.path.basename(os.path.normpath(sessionpath))


def get_sessions_list(path_to_sessions_folder, animal):
    def convert(func, *args):
        try:
            return func(*args)
        except ValueError:
            return False
        
    is_dir = lambda x: os.path.isdir(os.path.join(path_to_sessions_folder, x))
    has_parts = lambda x: len(x.split('_')) > 2
    starts_by_animal = lambda x: x.split('_')[0] == animal
    has_timestamp = lambda x: convert(datetime.strptime, '%s_%s' % (x.split('_')[-2], x.split('_')[-1]), '%Y-%m-%d_%H-%M-%S')

    return sorted([x for x in os.listdir(path_to_sessions_folder) if 
            is_dir(x) and has_parts(x) and starts_by_animal(x) and has_timestamp(x)])


def get_sampling_rate(sessionpath):
    filebase = os.path.basename(sessionpath)
    xml_path = os.path.join(sessionpath, filebase + '.xml')
    
    if not os.path.exists(xml_path):
        return None
    
    root = ET.parse(xml_path).getroot()
    sampling_rate = root.findall('acquisitionSystem')[0].findall('samplingRate')[0]
    return int(sampling_rate.text)


def unit_number_for_electrode(sessionpath, electrode_idx):
    filebase = ''
    try:
        filebase = guess_filebase(sessionpath)
    except ValueError:
        return 0  # no units on this electrode

    clu_file = os.path.join(sessionpath, '.'.join([filebase, 'clu', str(electrode_idx)]))
    cluster_map = np.loadtxt(clu_file, dtype=np.uint16)

    return len(np.unique(cluster_map)) - 1  # 1st cluster is noise


def unit_number_for_session(sessionpath):
    electrode_idxs = [x.split('.')[2] for x in os.listdir(sessionpath) if x.find('.clu.') > -1]

    idxs = []
    for el in electrode_idxs:
        try:
            elem = int(el)
            idxs.append(elem)
        except ValueError:
            pass

    unit_counts = [unit_number_for_electrode(sessionpath, el_idx) for el_idx in np.unique(idxs)]

    return np.array(unit_counts).sum()


def get_epochs(sessionpath):
    filebase = guess_filebase(sessionpath)
    h5name  = os.path.join(sessionpath, filebase + '.h5')

    with h5py.File(h5name, 'r') as f:
        cfg = json.loads(f['processed'].attrs['parameters'])

    tp = np.array(cfg['experiment']['timepoints'])
    s_duration = cfg['experiment']['session_duration']

    tp = np.repeat(tp, 2)
    epochs = np.concatenate([np.array([0]), tp, np.array([s_duration])])

    return epochs.reshape(int(len(epochs)/2), 2)


def cleaned_epochs(sessionpath):
    epochs = get_epochs(sessionpath)

    # if there are 5 epochs: session is continuous, remove speaker rotation periods
    if len(epochs) > 3:  
        epochs = epochs[slice(0, 5, 2)]

    # add whole session as epoch, if not yet there
    if len(epochs) > 1:
        whole_session = np.array([[epochs[0][0], epochs[-1][1]]])
        return np.concatenate([epochs, whole_session])
    else:
        return epochs