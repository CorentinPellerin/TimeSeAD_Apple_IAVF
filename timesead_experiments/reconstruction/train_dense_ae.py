import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.reconstruction import BasicAE
from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.optim.trainer import EarlyStoppingHook
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 12}},
        'reconstruction': {'class': 'ReconstructionTargetTransform', 'args': {'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 12}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True
    )

    split = (0.8, 0.2)


@training_ingredient.config
def training_config():
    loss = torch.nn.MSELoss
    batch_dim = get_batch_dim()
    trainer_hooks = [
        ('post_validation', EarlyStoppingHook)
    ]
    scheduler = {
        'class': torch.optim.lr_scheduler.MultiStepLR,
        'args': dict(milestones=[20], gamma=0.1)
    }


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        z_size=40,  # Size of the latent space (will be multiplied by window size)
    )

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    train_ds, val_ds = load_dataset()

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    training = Bunch(training)
    detector = MSEReconstructionAnomalyDetector(model).to(training.device)
    # detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    model_params = Bunch(model_params)
    ds_params = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = BasicAE(train_ds.num_features * train_ds.seq_len, model_params.z_size * train_ds.seq_len)

    trainer = train_model(_run, model, train_ds, val_ds)
    early_stop = trainer.hooks['post_validation'][-1]
    early_stop.get_best_epoch(True)
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
