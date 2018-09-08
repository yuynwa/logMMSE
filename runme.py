from scipy.io import wavfile
from scipy.special import expn
from scipy.fftpack import ifft
import numpy as np




def logMMSE(inputFilePath, outputFilePath):
    """
    %  Implements the logMMSE algorithm [1].

    Parameters
    ----------
    inputFilePath : string or open file handle
        Input wav file.

    outputFilePath: string or open file handle
        Output wav file.


    References
    ----------
    .. [1] Ephraim, Y. and Malah, D. (1985). Speech enhancement using a minimum
       mean-square error log-spectral amplitude estimator. IEEE Trans. Acoust.,
       Speech, Signal Process., ASSP-23(2), 443-445.

    """

    [sample_rate, sample_data] = wavfile.read(inputFilePath, True)

    # Frame size in samples
    len = np.int(np.floor(20 * sample_rate * 0.001))
    if len % 2 == 1:
        len += 1

    # window overlap in percent of frame size
    perc = 50
    len1 = np.floor(len * perc * 0.01)
    len2 = len - len1

    win = np.hanning(len)
    win = win * len2 / sum(win)

    # Noise magnitude calculations - assuming that the first 6 frames is noise / silence
    nFFT = len << 2
    noise_mean = np.zeros([nFFT, 1])
    dtype = 2 << 14
    j = 0


    for i in range(1, 7):

        s1 = j
        s2 = j + np.int(len)

        batch = sample_data[s1: s2] / dtype

        X = win * batch

        foo = np.fft.fft(X, np.int(nFFT))

        noise_mean += np.abs(foo.reshape(foo.shape[0], 1))

        j += len

    noise_mu = np.square(noise_mean / 6)

    # Allocate memory and initialize various variables

    x_old = np.zeros([np.int(len1), 1]);
    Nframes = np.floor(sample_data.shape[0] / len2) - np.floor(len / len2)
    xfinal = np.zeros([np.int(Nframes * len2), 1]);

    # Start Processing
    k = 0
    aa = 0.98
    mu = 0.98
    eta = 0.15

    ksi_min = 10 ** (-25 * 0.1)

    for n in range(0, np.int(Nframes)):

        s1 = k
        s2 = k + np.int(len)

        batch = sample_data[s1: s2] / dtype
        insign = win * batch

        spec = np.fft.fft(insign, nFFT)

        # Compute the magnitude
        sig = abs(spec)
        sig2 = sig ** 2

        # Limit post SNR to avoid overflows
        gammak = np.divide(sig2.reshape(sig2.shape[0], 1), noise_mu.reshape(noise_mu.shape[0], 1))
        gammak[gammak > 40] = 40

        foo = gammak - 1
        foo[foo < 0] = 0

        if 0 == n:
            ksi = aa + (1 - aa) * foo
        else:

            # a priori SNR
            ksi = aa * Xk_prev / noise_mu + (1 - aa) * foo

            # limit ksi to - 25 db
            ksi[ksi < ksi_min] = ksi_min

        log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
        vad_decision = sum(log_sigma_k) / len

        # noise only frame found
        if vad_decision < eta:
            noise_mu = mu * noise_mu + (1 - mu) * sig2.reshape([sig2.shape[0], 1])

        # == = end of vad == =

        # Log - MMSE estimator
        A = ksi / (1 + ksi)
        vk = A * gammak

        ei_vk = 0.5 * expn(1, vk)
        hw = A * np.exp(ei_vk)

        sig = sig.reshape([sig.shape[0], 1]) * hw
        Xk_prev = sig ** 2

        xi_w = ifft(hw * spec.reshape([spec.shape[0], 1]), nFFT, 0)
        xi_w = np.real(xi_w)

        xfinal[k: k + np.int(len2)] = x_old + xi_w[0: np.int(len1)]
        x_old = xi_w[np.int(len1): np.int(len)]

        k = k + np.int(len2)

    wavfile.write(outputFilePath, sample_rate, xfinal)

