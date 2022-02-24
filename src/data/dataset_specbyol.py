from torch.utils.data import Dataset
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
import torchaudio.transforms as T
from src.utils.interface_audio_augmentation import RunningNorm
import numpy as np
import random
import torch


def get_audio_file_path(file_list, index):
    audio_file = file_list[index]
    return audio_file[4:]


def load_waveform(audio_file, required_sampling_rate):
    waveform, sampling_rate = audio_io.audio_loader(audio_file)

    assert (
            sampling_rate == required_sampling_rate
    ), "sampling rate is not consistent throughout the dataset"
    return waveform


class SpectrogramDatasetWithWaveBYOL(Dataset):
    def __init__(self, file_path, audio_window=20480, sampling_rate=16000, augmentation=[1, 2, 3, 4, 5, 6],
                 config=None, augmentation_count=5):
        self.file_path = file_path
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.file_list = file_io.read_txt2list(self.file_path)
        self.config = config
        self.spectrogram = None
        if config['spectrogram_type'] == 'spectrogram':
            self.spectrogram = T.Spectrogram(
                n_fft=config['n_fft'],
                win_length=config['win_length'],
                hop_length=config['hop_length'],
                center=config['center'],
                pad_mode=config['pad_mode'],
                power=config['power']
            )
        elif config['spectrogram_type'] == 'mel_spectrogram':
            self.spectrogram = T.MelSpectrogram(
                sample_rate=config['sampling_rate'],
                n_fft=config['n_fft'],
                win_length=config['win_length'],
                hop_length=config['hop_length'],
                center=config['center'],
                pad_mode=config['pad_mode'],
                power=config['power'],
                norm=config['norm'],
                onesided=config['onesided'],
                n_mels=config['n_mels'],
                mel_scale=config['mel_scale']
            )
        elif config['spectrogram_type'] == 'mfcc':
            self.spectrogram = T.MFCC(
                sample_rate=config['sampling_rate'],
                n_mfcc=config['n_mfcc'],
                melkwargs={
                    'n_fft': config['n_fft'],
                    'n_mels': config['n_mels'],
                    'hop_length': config['hop_length'],
                    'mel_scale': config['mel_scale'],
                }
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = get_audio_file_path(self.file_list, index)

        waveform01 = audio_io.audio_adjust_length(load_waveform(audio_file, self.sampling_rate),
                                                  self.audio_window,
                                                  fit=False)
        waveform02 = audio_io.audio_adjust_length(load_waveform(audio_file, self.sampling_rate),
                                                  self.audio_window,
                                                  fit=False)

        pick_index = np.random.randint(waveform01.shape[1] - self.audio_window + 1)
        waveform01 = audio_io.random_cutoff(waveform01, self.audio_window, pick_index)
        waveform02 = audio_io.random_cutoff(waveform02, self.audio_window, pick_index)

        if len(self.augmentation) != 0:
            waveform01 = audio_augmentation.audio_augmentation_pipeline(waveform01, self.sampling_rate,
                                                                        self.audio_window,
                                                                        random.sample(self.augmentation,
                                                                                      self.augmentation_count),
                                                                        fix_audio_length=True)
            waveform02 = audio_augmentation.audio_augmentation_pipeline(waveform02, self.sampling_rate,
                                                                        self.audio_window,
                                                                        random.sample(self.augmentation,
                                                                                      self.augmentation_count),
                                                                        fix_audio_length=True)

        waveform01_spectrogram = self.spectrogram(waveform01)
        waveform02_spectrogram = self.spectrogram(waveform02)

        return waveform01_spectrogram, waveform02_spectrogram


class SpectrogramDatasetWithWaveBYOLTypeA(Dataset):
    def __init__(self, file_path, audio_window=20480, sampling_rate=16000, augmentation=[1, 2, 3, 4, 5, 6],
                 config=None, augmentation_count=5):
        self.file_path = file_path
        self.audio_window = audio_window
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.file_list = file_io.read_txt2list(self.file_path)
        self.config = config

        self.pre_norm = RunningNorm(epoch_samples=500)
        self.time_masking = T.TimeMasking(time_mask_param=8)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=8)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=config['sampling_rate'],
            n_fft=config['n_fft'],
            win_length=config['win_length'],
            hop_length=config['hop_length'],
            center=config['center'],
            pad_mode=config['pad_mode'],
            power=config['power'],
            norm=config['norm'],
            onesided=config['onesided'],
            n_mels=config['n_mels'],
            mel_scale=config['mel_scale']
        )
        self.mfcc = T.MFCC(
            sample_rate=config['sampling_rate'],
            n_mfcc=config['n_mfcc'],
            melkwargs={
                'n_fft': config['n_fft'],
                'n_mels': config['n_mels'],
                'hop_length': config['hop_length'],
                'mel_scale': config['mel_scale'],
            }
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_file = get_audio_file_path(self.file_list, index)

        waveform01 = audio_io.audio_adjust_length(load_waveform(audio_file, self.sampling_rate),
                                                  self.audio_window,
                                                  fit=False)
        waveform02 = audio_io.audio_adjust_length(load_waveform(audio_file, self.sampling_rate),
                                                  self.audio_window,
                                                  fit=False)

        pick_index = np.random.randint(waveform01.shape[1] - self.audio_window + 1)
        waveform01 = audio_io.random_cutoff(waveform01, self.audio_window, pick_index)
        waveform02 = audio_io.random_cutoff(waveform02, self.audio_window, pick_index)


        if len(self.augmentation) != 0:
            waveform01_a = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sampling_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
            waveform01_b = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sampling_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
            waveform01_c = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sampling_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)

            waveform02_a = audio_augmentation.audio_augmentation_pipeline(
                waveform02, self.sampling_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
            waveform02_b = audio_augmentation.audio_augmentation_pipeline(
                waveform02, self.sampling_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
            waveform02_c = audio_augmentation.audio_augmentation_pipeline(
                waveform02, self.sampling_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
        else:
            waveform01_a = waveform01.detach()
            waveform01_b = waveform01.detach()
            waveform01_c = waveform01.detach()
            waveform02_a = waveform02.detach()
            waveform02_b = waveform02.detach()
            waveform02_c = waveform02.detach()

        waveform01_a_spectrogram = self.mel_spectrogram(waveform01_a)
        waveform01_b_spectrogram = self.pre_norm((self.mel_spectrogram(waveform01_b) + torch.finfo().eps).log())
        waveform01_c_spectrogram = self.mfcc(waveform01_c)

        if len(self.augmentation) != 0:
            waveform01_a_spectrogram = self.time_masking(waveform01_a_spectrogram)
            waveform01_a_spectrogram = self.freq_masking(waveform01_a_spectrogram)
            waveform01_b_spectrogram = self.time_masking(waveform01_b_spectrogram)
            waveform01_b_spectrogram = self.freq_masking(waveform01_b_spectrogram)
            waveform01_c_spectrogram = self.time_masking(waveform01_c_spectrogram)
            waveform01_c_spectrogram = self.freq_masking(waveform01_c_spectrogram)

        waveform02_a_spectrogram = self.mel_spectrogram(waveform02_a)
        waveform02_b_spectrogram = self.pre_norm((self.mel_spectrogram(waveform02_b) + torch.finfo().eps).log())
        waveform02_c_spectrogram = self.mfcc(waveform02_c)

        if len(self.augmentation) != 0:
            waveform02_a_spectrogram = self.time_masking(waveform02_a_spectrogram)
            waveform02_a_spectrogram = self.freq_masking(waveform02_a_spectrogram)
            waveform02_b_spectrogram = self.time_masking(waveform02_b_spectrogram)
            waveform02_b_spectrogram = self.freq_masking(waveform02_b_spectrogram)
            waveform02_c_spectrogram = self.time_masking(waveform02_c_spectrogram)
            waveform02_c_spectrogram = self.freq_masking(waveform02_c_spectrogram)


        waveform01_spectrogram = torch.stack([waveform01_a_spectrogram[0], waveform01_b_spectrogram[0], waveform01_c_spectrogram[0]])
        waveform02_spectrogram = torch.stack([waveform02_a_spectrogram[0], waveform02_b_spectrogram[0], waveform02_c_spectrogram[0]])

        return waveform01_spectrogram, waveform02_spectrogram