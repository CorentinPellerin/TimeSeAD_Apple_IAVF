import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.reconstruction import MSCREDLoss, MSCRED, MSCREDAnomalyDetectorOrig
from timesead.optim.trainer import EarlyStoppingHook
from timesead.utils.utils import Bunch, objspec2constructor


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'timesead.models.reconstruction.SignatureMatrixTransform', 'args': {'wins': (10, 30, 60),
                                                                                       'seg_interval': 10, 'h': 5}},
        'reconstruction': {'class': 'ReconstructionTargetTransform', 'args': {'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'timesead.models.reconstruction.SignatureMatrixTransform', 'args': {'wins': (10, 30, 60),
                                                                                       'seg_interval': 10, 'h': 5}},
    }


def get_batch_dim():
    return 1


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True
    )

    split = (0.8, 0.2)


@training_ingredient.config
def training_config():
    batch_dim = get_batch_dim()
    loss = MSCREDLoss
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
        c_out=256,
        small_model=False,
        chi=5.0
    )
    detector = {
        'class': MSCREDAnomalyDetectorOrig,
        'args': dict(
            error_threshold=0.5
        )
    }

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    train_ds, val_ds = load_dataset()

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, detector, _run, save_detector=True):
    training = Bunch(training)
    detector = objspec2constructor(detector)(model).to(training.device)
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
    model = MSCRED(train_ds.num_features[1], train_ds.num_features[0], **model_params._adict)

    trainer = train_model(_run, model, train_ds, val_ds)
    early_stop = trainer.hooks['post_validation'][-1]
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
