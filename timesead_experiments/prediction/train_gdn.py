import torch

from timesead.models.prediction import GDN, TCNPredictionAnomalyDetector
from timesead.optim.trainer import EarlyStoppingHook
from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'prediction': {'class': 'PredictionTargetTransform', 'args': {'window_size': 5, 'prediction_horizon': 1,
                                                                      'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'prediction': {'class': 'PredictionTargetTransform', 'args': {'window_size': 5, 'prediction_horizon': 1,
                                                                      'replace_labels': False}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True,
    )

    split = (0.8, 0.2)


@training_ingredient.config
def training_config():
    batch_dim = get_batch_dim()
    loss = torch.nn.MSELoss
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
        dim=64,
        out_layer_hidden_dims=(64,),
        topk=15,
        dropout_prob=0.2
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

    # if val_loader is None:
    #     train_ds, val_ds = load_dataset()
    #     # Train for 0 epochs to get the val loader
    #     trainer = train_model(_run, model, train_ds, val_ds, epochs=0)
    #     val_loader = trainer.val_iter

    detector = TCNPredictionAnomalyDetector(model).to(training.device)
    # detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    ds_params = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = GDN(train_ds.num_features, train_ds.seq_len, **model_params)

    trainer = train_model(_run, model, train_ds, val_ds)
    early_stop = trainer.hooks['post_validation'][-1]
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
