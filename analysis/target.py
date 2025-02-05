import numpy as np


def build_tgt_matrix(tl, trials, aeps_events):
    # compute timeline / AEP indices of entrances / exist to the target
    tl_tgt_start_idxs   = []  # timeline indices of first target pulse
    tl_tgt_end_idxs     = []  # timeline indices of last target pulse
    aeps_tgt_start_idxs = []  # indices of first AEPs in target
    aeps_tgt_end_idxs   = []  # indices of last AEPs in target
    
    for i in range(len(tl) - 1):
        if tl[i][6] < 2 and tl[i+1][6] == 2:
            nearest_aep_idx = np.abs(aeps_events[:, 0] - tl[i+1][0]).argmin()
            aeps_tgt_start_idxs.append(nearest_aep_idx)
            t_event = aeps_events[nearest_aep_idx][0]
            tl_tgt_start_idxs.append(np.abs(tl[:, 0] - t_event).argmin())
        if tl[i][6] == 2 and tl[i+1][6] < 2:
            nearest_aep_idx = np.abs(aeps_events[:, 0] - tl[i][0]).argmin()
            aeps_tgt_end_idxs.append(nearest_aep_idx)
            t_event = aeps_events[nearest_aep_idx][0]
            tl_tgt_end_idxs.append(np.abs(tl[:, 0] - t_event).argmin())
            
    # ignore first/last target if not ended
    if tl_tgt_start_idxs[-1] > tl_tgt_end_idxs[-1]:
        tl_tgt_start_idxs = tl_tgt_start_idxs[:-1]
        aeps_tgt_start_idxs = aeps_tgt_start_idxs[:-1]
    if tl_tgt_end_idxs[0] < tl_tgt_start_idxs[0]:
        tl_tgt_end_idxs = tl_tgt_end_idxs[1:]
        aeps_tgt_end_idxs = aeps_tgt_end_idxs[1:]
    tl_tgt_start_idxs = np.array(tl_tgt_start_idxs)
    tl_tgt_end_idxs   = np.array(tl_tgt_end_idxs)

    # successful / missed
    tgt_results = np.zeros(len(tl_tgt_start_idxs))
    for idx_tl_success_end in trials[trials[:, 5] == 1][:, 1]:
        idx_succ = np.abs(tl_tgt_end_idxs - idx_tl_success_end).argmin()
        tgt_results[idx_succ] = 1

    # tl_idx_start, tl_idx_end, aep_idx_start, aer_idx_end, success / miss
    return np.column_stack([
        tl_tgt_start_idxs,
        tl_tgt_end_idxs,
        aeps_tgt_start_idxs,
        aeps_tgt_end_idxs,
        tgt_results
    ]).astype(np.int32)


def build_silence_matrix(tl):
    idxs_silence_start, idxs_silence_end = [], []
    for i in range(len(tl) - 1):
        if tl[i][6] != 0 and tl[i+1][6] == 0:    # silence start
            idxs_silence_start.append(i+1)
        elif tl[i][6] == 0 and tl[i+1][6] != 0:  # silence end
            idxs_silence_end.append(i)

    if len(idxs_silence_start) > len(idxs_silence_end):
        idxs_silence_start = idxs_silence_start[:-1]
    idxs_silence_start = np.array(idxs_silence_start)
    idxs_silence_end   = np.array(idxs_silence_end)

    return np.column_stack([idxs_silence_start, idxs_silence_end])


def get_idxs_of_event_periods(tl, event_type):
    # event_type: -1, 0, 1, 2 (noise, silence, background, target)
    # returns: indices to timeline for periods of event_type
    idxs_events  = np.where(tl[:, 6] == event_type)[0]
    idxs_to_idxs = np.where(np.diff(idxs_events) > 1)[0]

    # periods - indices to TL where was silent
    periods       = np.zeros([len(idxs_to_idxs) + 1, 2])
    periods[0]    = np.array([0, idxs_to_idxs[0]])
    periods[1:-1] = np.column_stack([idxs_to_idxs[:-1] + 1, idxs_to_idxs[1:]])
    periods[-1]   = np.array([idxs_to_idxs[-1], len(idxs_events) - 1])
    periods       = periods.astype(np.int32)

    # convert to TL indices
    return np.column_stack([idxs_events[periods[:, 0]], idxs_events[periods[:, 1]]])


def build_silence_and_noise_events(tl, offset, latency, drift):
    # build hallucination pulses in silence and noise
    duration = tl[-1][0]
    
    # all pulses with drift
    #pulse_times = np.linspace(0, int(duration - latency), int(duration - latency)*4 + 1) + offset # if latency 0.25
    pulse_times = np.array([i*latency for i in range(int((duration - latency)/latency) + 10)]) + offset
    pulse_times = pulse_times[pulse_times < duration]
    pulse_times += np.arange(len(pulse_times)) * drift/len(pulse_times)

    # filter silence times only
    pulses_silence = []
    pulses_noise   = []
    pulses_bgr     = []
    pulses_tgt     = []
    tl_idx = 0  # index of current pulse in the timeline
    for t_pulse in pulse_times:
        while tl[tl_idx][0] < t_pulse:
            tl_idx += 1

        if tl[tl_idx][6] == 0:
            pulses_silence.append(t_pulse)
        elif tl[tl_idx][6] == -1:
            pulses_noise.append(t_pulse)
        elif tl[tl_idx][6] == 1:
            pulses_bgr.append(t_pulse)
        elif tl[tl_idx][6] == 2:
            pulses_tgt.append(t_pulse)
            
    pulses_silence = np.array(pulses_silence)
    pulses_noise   = np.array(pulses_noise)
    pulses_bgr   = np.array(pulses_bgr)
    pulses_tgt   = np.array(pulses_tgt)
    return pulses_silence, pulses_noise, pulses_bgr, pulses_tgt


def build_event_mx(tl, offset, latency):
    drift_coeff = 0.055/2400
    duration = tl[-1][0]
    drift = duration * drift_coeff
    
    # all pulses with drift
    pulse_times = np.array([i*latency for i in range(int((duration - latency)/latency) + 10)]) + offset
    pulse_times += np.arange(len(pulse_times)) * drift/len(pulse_times)
    pulse_times = pulse_times[pulse_times < duration]  # filter out if more pulses
    
    event_mx = np.zeros([len(pulse_times), 2])
    tl_idx = 0  # index of current pulse in the timeline
    for i, t_pulse in enumerate(pulse_times):
        while tl[tl_idx][0] < t_pulse:
            tl_idx += 1

        event_mx[i] = np.array([t_pulse, tl[tl_idx][6]])
    return event_mx[:-1]


def get_spike_times_at(tl, s_times, periods, mode='sequence'):
    # 'sequence' - periods follow each other in a sequence
    # 'overlay'  - all periods aligned to time zero
    all_spikes  = []  # collect as groups
    sil_dur = 0
    for period in periods:
        idxs_tl_l, idxs_tl_r = period[0], period[1]

        spikes = s_times[(s_times > tl[idxs_tl_l][0]) & (s_times < tl[idxs_tl_r][0])]
        spikes -= tl[idxs_tl_l][0]  # align to time 0
        if mode == 'sequence':
            spikes += sil_dur  # adjust to already processed silence periods
        all_spikes.append(spikes)

        sil_dur += tl[idxs_tl_r][0] - tl[idxs_tl_l][0]
    return all_spikes  #np.array([item for sublist in all_spikes for item in sublist])


def get_spike_counts(spk_times, pulse_times, hw=0.25, bin_count=51):
    collected = []
    for t_pulse in pulse_times:
        selected = spk_times[(spk_times > t_pulse - hw) & (spk_times < t_pulse + hw)]
        collected += [x for x in selected - t_pulse]
    collected = np.array(collected)

    bins = np.linspace(-hw, hw, bin_count)
    counts, _ = np.histogram(collected, bins=bins)
    counts = counts / len(pulse_times) # * 1/((2. * hw)/float(bin_count - 1))
    counts = counts / (bins[1] - bins[0])  # divide by bin size to get firing rate
    
    return bins, counts