import numpy as np
import torch
from whisper.audio import N_FFT, N_MELS, mel_filters, HOP_LENGTH

def log_mel_spectrogram(audio: np.ndarray):
    audio = torch.from_numpy(audio)
    audio = audio.to("cpu")

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, N_MELS)
    mel_spec = filters @ magnitudes
    # return torch.clamp(mel_spec, min=1e-10).log10()
    
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()+10
    log_spec_min = log_spec.max() - 8.0
    log_spec = torch.maximum(log_spec, log_spec_min)
    log_spec = (log_spec - log_spec.min())
    return log_spec/8