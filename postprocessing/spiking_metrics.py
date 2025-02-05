import numpy as np

# See also: https://github.com/NeuralEnsemble/NeuroTools/blob/master/src/signals/spikes.py


def mean_firing_rate(spiketrain):
    """ 
    spiketrain - an array of spike times in seconds
    """
    if len(spiketrain) < 5:
        return 0
    return np.mean(np.diff(spiketrain)) ** (-1)


def isi_cv(spiketrain, outliers=3):
    """ 
    ISI Coeff of variation
    
    spiketrain - an array of spike times in seconds
    """
    if len(spiketrain) < 5:
        return 0
    isi = np.diff(spiketrain)
    #isi = isi[abs(isi - np.mean(isi)) < outliers * np.std(isi)]
    return np.std(isi)/np.mean(isi)


def isi_fano(spiketrain, outliers=3):
    """ 
    The Fano Factor is defined as the variance of the isi divided by the mean of the isi
    
    http://en.wikipedia.org/wiki/Fano_factor
    From https://github.com/NeuralEnsemble/NeuroTools/blob/master/src/signals/spikes.py
    
    spiketrain - an array of spike times in seconds
    """
    if len(spiketrain) < 5:
        return 0
    isi = np.diff(spiketrain)
    #isi_filt = isi[abs(isi - np.mean(isi)) < outliers * np.std(isi)]
    return np.var(isi)/np.mean(isi)


def burstiness(spiketrain, threshold=0.02):
    """
    spiketrain - an array of spike times in seconds
    threshold  - 
    """
    pass  # Not implemented