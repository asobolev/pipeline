import numpy as np
from scipy import signal


def head_direction(tl, hd_update_speed=0.04):
    width = 200  # 100 points ~= 1 sec with at 100Hz
    kernel = signal.gaussian(width, std=(width) / 7.2)

    x_smooth = np.convolve(tl[:, 1], kernel, 'same') / kernel.sum()
    y_smooth = np.convolve(tl[:, 2], kernel, 'same') / kernel.sum()

    diff_x = np.diff(x_smooth, axis=0)
    diff_y = np.diff(y_smooth, axis=0)

    hd = -np.arctan2(diff_y, diff_x)
    hd = np.concatenate([np.array([hd[0]]), hd])  # make the same length as timeline
    
    # reset idle periods
    idle_periods = []
    idle_idxs = np.where(tl[:, 3] < hd_update_speed)[0]

    crit = np.where(np.diff(idle_idxs) > 1)[0]

    idle_periods.append( (idle_idxs[0], idle_idxs[crit[0]]) )  # first idle period
    for i, point in enumerate(crit[:-1]):
        idx_start = idle_idxs[crit[i] + 1]
        idx_end = idle_idxs[crit[i+1]]
        idle_periods.append( (idx_start, idx_end) )

    idle_periods = np.array(idle_periods)
    
    for (i1, i2) in idle_periods:
        hd[i1:i2] = hd[i1-1]
        
    return hd


def head_direction_slow(tl):  # not used, slow computation
    offset = 10
    hd_update_speed = 0.04
    hd = np.zeros(len(tl))

    for i, pos in enumerate(tl[offset:]):
        recent_traj = tl[(tl[:, 0] > pos[0] - 0.25) & (tl[:, 0] <= pos[0])]
        avg_speed = recent_traj[:, 3].mean()
        if avg_speed > hd_update_speed:  # if animal runs basically
            x, y = recent_traj[0][1], recent_traj[0][2]
            vectors = [np.array([a[1], a[2]]) - np.array([x, y]) for a in recent_traj[1:]]
            avg_direction = np.array(vectors).sum(axis=0) / len(vectors)

            avg_angle = -np.arctan2(avg_direction[1], avg_direction[0])
            hd[i + offset] = avg_angle  # in radians
        else:
            hd[i + offset] = hd[i + offset - 1]

    first_non_zero_idx = np.nonzero(hd)[0][0]
    hd[:first_non_zero_idx] = hd[first_non_zero_idx]
    
    return hd