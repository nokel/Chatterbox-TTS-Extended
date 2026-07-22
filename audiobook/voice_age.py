import numpy as np
import librosa
import scipy.fft

N_FFT = 1024
HOP = 256


def _envelope(logmag, lifter=24):
    c = scipy.fft.dct(logmag, type=2, axis=0, norm="ortho")
    c[lifter:, :] = 0.0
    return scipy.fft.idct(c, type=2, axis=0, norm="ortho")


def formant_shift(x, sr, ratio):
    if abs(ratio - 1.0) < 1e-3:
        return x
    S = librosa.stft(x, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(S), np.angle(S)
    logmag = np.log(mag + 1e-8)
    env = _envelope(logmag)
    F = env.shape[0]
    src = np.arange(F)
    warped = np.empty_like(env)
    for t in range(env.shape[1]):
        warped[:, t] = np.interp(src / ratio, src, env[:, t],
                                 left=env[0, t], right=env[-1, t])
    new_mag = np.exp(logmag - env + warped)
    y = librosa.istft(new_mag * np.exp(1j * phase), hop_length=HOP,
                      length=len(x))
    return y.astype(np.float32)


def pitch_shift(x, sr, semitones):
    if abs(semitones) < 1e-3:
        return x
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones)


def deage(x, sr, pitch_semitones=3.0, formant_ratio=1.15):
    y = pitch_shift(x, sr, pitch_semitones)
    corrective = formant_ratio / (2.0 ** (pitch_semitones / 12.0))
    y = formant_shift(y, sr, corrective)
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def deage_file(in_path, out_path, pitch_semitones=3.0, formant_ratio=1.15):
    import soundfile as sf
    x, sr = librosa.load(in_path, sr=None, mono=True)
    y = deage(x, sr, pitch_semitones, formant_ratio)
    sf.write(out_path, y, sr)
    return out_path
