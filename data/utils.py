import numpy as np

# this function adds guassian noise to an input signal
# the center of the noise is 0
# the std dev of the nosie is alpha*std_dev of the signal
def add_guassian_noise(signal, alpha=0.15):
    assert signal.ndim == 1
    return signal + np.random.normal(0,alpha*np.std(signal),len(signal))

def augment_patient(patient):
    ret = patient.copy()
    # first is the time which we do not add noise to
    for signal in patient.columns[1:]:
        # dont add noise to the label
        if signal == 'label':
            continue
        ret[signal] = add_guassian_noise(patient[signal])
    assert ret.shape == patient.shape
    return ret

