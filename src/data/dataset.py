import src.data.dataset_specbyol as dataset_specbyol
import src.data.dataset_urbansound as dataset_urbansound
import src.data.dataset_voxceleb as dataset_voxceleb
from torch.utils import data


def get_dataloader(config, mode="train"):
    dataset_type = config['dataset_type']
    dataset = None

    if dataset_type == 'WaveformDatasetByWaveBYOL':
        dataset = dataset_specbyol.SpectrogramDatasetWithWaveBYOL(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count'],
            config=config
        )
    elif dataset_type == 'WaveformDatasetByWaveBYOLTypeA':
        dataset = dataset_specbyol.SpectrogramDatasetWithWaveBYOLTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sampling_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count'],
            config=config
        )
    elif dataset_type == 'UrbanSound8KSpecDatasetTypeA':
        dataset = dataset_urbansound.UrbanSound8KSpecDatasetTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sample_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count'],
            config=config
        )
    elif dataset_type == 'VoxCelebSpecDatasetTypeA':
        dataset = dataset_voxceleb.VoxCelebSpecDatasetTypeA(
            file_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            sample_rate=config['sampling_rate'],
            augmentation=config['{}_augmentation'.format(mode)],
            augmentation_count=config['augmentation_count'],
            config=config
        )


    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=config['dataset_shuffle'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )

    return dataloader, dataset
