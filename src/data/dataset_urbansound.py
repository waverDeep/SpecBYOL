from torch.utils.data import Dataset
import src.utils.interface_file_io as file_io
import src.utils.interface_audio_io as audio_io
import src.utils.interface_audio_augmentation as audio_augmentation
import torchaudio.transforms as T
from src.utils.interface_audio_augmentation import RunningNorm
import numpy as np
import random
import torch
import pandas as pd
import natsort


def get_acoustic_dict(acoustic_list):
    acoustic_dict = {}
    for idx, key in enumerate(acoustic_list):
        acoustic_dict[str(key)] = idx
    return acoustic_dict


def get_audio_file_path(file_list, index):
    audio_file = file_list[index]
    return audio_file[4:]


def get_audio_file_with_acoustic_info(file_list, index):
    audio_file = get_audio_file_path(file_list, index)
    filename = audio_file.split('/')[5]
    acoustic_id = filename.split('-')[1]
    return audio_file, filename, acoustic_id


def search_cutting_boundary(metadata, filename):
    line = metadata[metadata['slice_file_name'] == filename]
    bound = [float(line['start']), float(line['end'])]
    return bound


def load_waveform(audio_file, required_sampling_rate):
    waveform, sampling_rate = audio_io.audio_loader(audio_file)

    assert (
            sampling_rate == required_sampling_rate
    ), "sampling rate is not consistent throughout the dataset"
    return waveform


def load_data_pipeline(audio_file, required_sample_rate, audio_window, full_audio, augmentation, cut_silence=None, custom_augmentation_list=None):
    waveform, sample_rate = audio_io.audio_loader("{}".format(audio_file))

    if cut_silence is not None:
        waveform = audio_io.cutoff(waveform, sample_rate, cut_silence[0], cut_silence[1])

    assert (
            sample_rate == required_sample_rate
    ), "sampling rate is not consistent throughout the dataset"
    waveform = audio_io.audio_adjust_length(waveform, audio_window)

    if not full_audio:
        waveform = audio_io.random_cutoff(waveform, audio_window)
    if augmentation:
        waveform = audio_augmentation.audio_augmentation_baseline(waveform, sample_rate, audio_window,
                                                                  custom_augmentation_list=custom_augmentation_list)
    if not full_audio:
        waveform = audio_io.audio_adjust_length(waveform, audio_window)
    return waveform


class UrbanSound8KSpecDatasetTypeA(Dataset):
    def __init__(self, file_path: str, audio_window=20480, sample_rate=16000, full_audio=False, augmentation=False,
                 metadata="./dataset/UrbanSound8K/metadata/UrbanSound8K.csv", config=None, augmentation_count=None):
        super(UrbanSound8KSpecDatasetTypeA, self).__init__()
        self.file_path = file_path
        self.audio_window = audio_window
        self.sample_rate = sample_rate
        self.full_audio = full_audio
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count

        # data file list
        id_data = open(self.file_path, 'r')
        self.file_list = [x.strip() for x in id_data.readlines()]
        id_data.close()

        self.augmentation = augmentation
        self.metadata = None
        if metadata is not None:
            self.metadata = pd.read_csv(metadata)
        self.acoustic_list = natsort.natsorted(list(set(self.metadata['classID'])))
        self.acoustic_dict = get_acoustic_dict(self.acoustic_list)

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
        audio_file, filename, acoustic_id = get_audio_file_with_acoustic_info(self.file_list, index)

        waveform01 = audio_io.audio_adjust_length(load_waveform(audio_file, self.sample_rate),
                                                  self.audio_window,
                                                  fit=False)

        pick_index = np.random.randint(waveform01.shape[1] - self.audio_window + 1)
        waveform01 = audio_io.random_cutoff(waveform01, self.audio_window, pick_index)

        if len(self.augmentation) != 0:
            waveform01_a = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sample_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
            waveform01_b = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sample_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)
            waveform01_c = audio_augmentation.audio_augmentation_pipeline(
                waveform01, self.sample_rate,
                self.audio_window,
                random.sample(self.augmentation, random.randint(1, self.augmentation_count)),
                fix_audio_length=True)

        else:
            waveform01_a = waveform01.detach()
            waveform01_b = waveform01.detach()
            waveform01_c = waveform01.detach()

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

        waveform01_spectrogram = torch.stack(
            [waveform01_a_spectrogram[0], waveform01_b_spectrogram[0], waveform01_c_spectrogram[0]])

        return waveform01_spectrogram, str(acoustic_id)