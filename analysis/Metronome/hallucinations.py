import os
import numpy as np
import h5py


def get_pop_resp_profile_mx(source, session, unit_names, hw, bc):
    animal      = session.split('_')[0]
    h5_file     = os.path.join(source, animal, session, session + '.h5')
            
    bins = np.linspace(-hw, hw, bc)
    profile_mx = np.zeros([len(unit_names), bc - 1])
    for i, unit_name in enumerate(unit_names):
        with h5py.File(h5_file, 'r') as f:
            shuffled = np.array(f['units'][unit_name]['psth_shuffled_micro_in_silence'])
            profiles = np.array(f['units'][unit_name]['psth_profiles_silence'])

        fr_mean = shuffled[:, 1]
        fr_std  = shuffled[:, 2]
        fr_prof = profiles.mean(axis=0)
        profile_mx[i] = (fr_prof - fr_mean)/fr_std  # z-scored
        
    return profile_mx, bins