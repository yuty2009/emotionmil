# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as signal


def bandpower(x, fs, bands, method='fft'):
    """ Compute power in each frequency bin specified by bands from FFT result of
        x. By default, x is a real signal.
        Refer to https://github.com/forrestbao/pyeeg/blob/master/pyeeg/spectrum.py

        Parameters
        -----------
        x
            list
            a 1-D real time series.
        fs
            integer
            the sampling rate in physical frequency.
        bands
            list
            boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
            [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
            You can also use range() function of Python to generate equal bins and
            pass the generated list to this function.
            Each element of Band is a physical frequency and shall not exceed the
            Nyquist frequency, i.e., half of sampling frequency.

        method
            string
            power estimation method fft/burg/welch, default fft.

        Returns
        -------
        power
            list
            spectral power in each frequency bin.
        """

    L = len(x)
    if method == 'fft':
        pxx = np.fft.fft(x)
        pxx = abs(pxx)
        pxx = pxx[:L//2]
        f = np.arange(L / 2) / L
    elif method == 'welch':
        f, pxx = signal.welch(x, fs)
    elif method == 'periodogram':
        f, pxx = signal.periodogram(x, fs)
    else:
        assert 'unknown method'

    num_bands = len(bands) - 1
    pxx_bands = np.zeros(num_bands)
    for i in range(0, num_bands):
        f1 = float(bands[i]) / fs
        f2 = float(bands[i + 1]) / fs
        indices = np.argwhere((f >= f1) & (f < f2))
        pxx_bands[i] = sum(pxx[indices])
    return pxx_bands


def differentialentropy(x, fs, bands, order=5):
    """ Compute differential entropy from 4 bands (theta, alpha, beta, gamma) filtered signal
    x. By default, x is a real signal.
    - Related Paper: Fdez J, Guttenberg N, Witkowski O, et al. Cross-subject EEG-based emotion recognition through neural networks with stratified normalization[J]. Frontiers in neuroscience, 2021, 15: 626277.
    - Related Project: https://github.com/javiferfer/cross-subject-eeg-emotion-recognition-through-nn/

    - Related Paper: Li D, Xie L, Chai B, et al. Spatial-frequency convolutional self-attention network for EEG emotion recognition[J]. Applied Soft Computing, 2022, 122: 108740.
    - Related Project: https://github.com/qeebeast7/SFCSAN/
    """
    de = np.zeros(len(bands)-1)
    for i in range(len(bands)-1):
        f1, f2 = bands[i], bands[i+1]
        b, a = signal.butter(order, [f1, f2], btype='bandpass', fs=fs)
        y = signal.lfilter(b, a, x)
        de[i] = 1 / 2 * np.log2(2 * np.pi * np.e * np.std(y))
    return de


def differentialentropy1(x, fs, bands, nfft=None):
    """ Compute differential entropy in each frequency bin specified by bands from FFT result of
    x. By default, x is a real signal.
    """
    L = len(x)
    if nfft is None:
        nfft = L
    hann = signal.windows.hann(L)
    x = x * hann
    y = np.fft.fft(x, nfft)
    pxx = np.abs(y[:nfft//2])
    de = np.zeros(len(bands)-1)
    for i in range(len(bands)-1):
        f1, f2 = bands[i], bands[i+1]
        idx1 = int(np.floor(nfft * f1 / fs))
        idx2 = int(np.floor(nfft * f2 / fs))
        band_ave_psd = np.mean(pxx[idx1-1:idx2]**2)
        de[i] = np.log2(100 * band_ave_psd)
    return de


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fs = 1000
    T = 1 / fs
    L = 1500
    t = np.arange(L) * T
    f1 = np.arange(L / 2) / L
    nfft = int(2 ** np.ceil(np.log2(L)))
    s = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    x = s + 2 * np.random.randn(t.size)

    y = np.fft.fft(x)
    pxx1 = abs(y)
    pxx1 = pxx1[:L//2]
    f2, pxx2 = signal.periodogram(x)
    pxx2[0] = np.median(pxx2[:3])
    f3, pxx3 = signal.welch(x)
    """
    plt.subplot(411)
    plt.plot(t * fs, x)
    plt.title('Signal Corrupted with Zero-Mean Random Noise')
    plt.xlabel('t (milliseconds)')
    plt.ylabel('X(t)')

    plt.subplot(412)
    plt.plot(f1 * fs, 10 * np.log10(pxx1))
    plt.title('Single-Sided Amplitude Spectrum of X(t)')
    plt.xlabel('f (Hz)')
    plt.ylabel('10*log10|P1(f)|')

    plt.subplot(413)
    plt.plot(f2, 10 * np.log10(pxx2))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power (dB)')
    plt.title('Periodogram')

    plt.subplot(414)
    plt.plot(f3, 10 * np.log10(pxx3))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power (dB)')
    plt.title('Welchs')

    plt.show()
    """

    from transforms import BandDifferentialEntropyV1

    fs = 128
    n_channels = 2
    x = np.random.randn(n_channels, 128)
    bands = [4, 8, 14, 31, 49]
    bde = BandDifferentialEntropyV1()
    de1 = bde(eeg=x)
    de1 = de1['eeg']
    de2 = np.zeros((n_channels, len(bands)-1))
    for i in range(n_channels):
        de2[i] = differentialentropy1(x[i, :], fs, bands)
    print(de1)
    print(de2)
