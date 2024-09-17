import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from matplotlib.colors import SymLogNorm
from torch import nn, binary_cross_entropy_with_logits
from torchvision import models


def get_rms(data: np.ndarray, maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS

    :param data: numpy array
    :param maskSup: mask threshold

    :return: rms --> rms of image
    """

    mIn = np.ndarray.flatten(data)
    m = mIn[np.abs(mIn) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)

    for i in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold) / rmsold) < diff:
            break
        rmsold = rms

    return rms  # jy/beam


def normalize_fits(image_data: np.ndarray):
    image_data = image_data.squeeze()

    # Pre-processing
    rms = get_rms(image_data)
    norm = SymLogNorm(linthresh=rms * 2, linscale=2, vmin=-rms, vmax=rms * 50000, base=10)

    image_data = norm(image_data)
    image_data = np.clip(image_data - image_data.min(), a_min=0, a_max=1)

    # make RGB image
    cmap = plt.get_cmap('RdBu_r')
    image_data = cmap(image_data)
    image_data = np.delete(image_data, 3, 2)

    image_data = -image_data + 1  # make the peak exist at zero

    return image_data


def process_fits(fits_path):
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data

    return normalize_fits(image_data)


PROFILE = False
SEED = None


def init_vit(model_name):
    assert model_name == 'vit_l_16'

    backbone = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
    for param in backbone.parameters():
        param.requires_grad_(False)

    # backbone.class_token[:] = 0
    backbone.class_token.requires_grad_(True)

    hidden_dim = backbone.heads[0].in_features

    del backbone.heads

    return backbone, hidden_dim


def init_cnn(name: str):
    # use partial to prevent loading all models at once
    model_map = {
        'resnet50': partial(models.resnet50, weights="DEFAULT"),
        'resnet152': partial(models.resnet152, weights="DEFAULT"),
        'resnext50_32x4d': partial(models.resnext50_32x4d, weights="DEFAULT"),
        'resnext101_64x4d': partial(models.resnext101_64x4d, weights="DEFAULT"),
        'efficientnet_v2_l': partial(models.efficientnet_v2_l, weights="DEFAULT"),
    }

    backbone = model_map[name]()

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    num_out_features = (
        backbone.fc if name in ('resnet50', 'resnet152', 'resnext50_32x4d', 'resnext101_64x4d')
        else backbone.classifier[-1]  # efficientnet
    ).in_features

    return feature_extractor, num_out_features


def get_classifier(dropout_p: float, n_features: int, num_target_classes: int):
    assert 0 <= dropout_p <= 1

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout1d(p=dropout_p),
        nn.Linear(n_features, n_features),
        nn.ReLU(),
        nn.Linear(n_features, num_target_classes),
    )

    return classifier


@torch.no_grad()
def normalize_inputs(inputs, mode=0):
    assert mode in range(3)

    if mode == 0:
        # Actual op instead of simple return because of debugging reasons
        means, stds = [0, 0, 0], [1, 1, 1]
    elif mode == 1:
        # Inputs are lognormal -> log to make normal
        means, stds = [-1.55642344, -1.75137082, -2.13795913], [1.25626133, 0.79308821, 0.7116124]
        inputs = inputs.log()
    else:
        # Inputs are lognormal
        means, stds = [0.35941373, 0.23197646, 0.15068751], [0.28145176, 0.17234328, 0.10769559]

    # Resulting shape of means and stds: [1, 3, 1, 1]
    means, stds = map(
        lambda x: torch.tensor(x, device=inputs.device).reshape(1, 3, 1, 1),
        (means, stds)
    )

    return (inputs - means) / stds


class ImagenetTransferLearning(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet50',
        dropout_p: float = 0.25,
        use_compile: bool = True
    ):
        super().__init__()

        get_classifier_f = partial(get_classifier, dropout_p=dropout_p, num_target_classes=1)

        # For saving in the state dict
        self.kwargs = {'model_name': model_name, 'dropout_p': dropout_p}

        if model_name == 'vit_l_16':
            self.vit, num_features = init_vit(model_name)
            self.vit.eval()

            classifier = get_classifier_f(n_features=num_features)
            self.vit.heads = classifier

            self.forward = self.vit_forward

        else:
            self.feature_extractor, num_features = init_cnn(name=model_name)
            self.feature_extractor.eval()

            self.classifier = get_classifier_f(n_features=num_features)

            self.forward = self.cnn_forward

        if use_compile:
            self.forward = torch.compile(model=self.forward, mode='reduce-overhead')

    # @partial(torch.compile, mode='reduce-overhead')
    def cnn_forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x

    # @partial(torch.compile, mode='reduce-overhead')
    def vit_forward(self, x):

        x = self.vit.forward(x)

        return x

    def step(self, inputs, targets):
        logits = self(inputs).flatten()

        loss = binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.as_tensor(2.698717948717949)
        )

        if PROFILE:
            global profiler
            profiler.step()

        return logits, loss

    def eval(self):
        if self.kwargs['model_name'] == 'vit_l_16':
            self.vit.heads.eval()
        else:
            self.classifier.eval()

    def train(self):
        if self.kwargs['model_name'] == 'vit_l_16':
            self.vit.heads.train()
        else:
            self.classifier.train()


def load_checkpoint(ckpt_path, device="gpu"):
    if os.path.isfile(ckpt_path):
        ckpt_dict = torch.load(ckpt_path, weights_only=False, map_location=device)
    else:
        files = os.listdir(ckpt_path)
        possible_checkpoints = list(filter(lambda x: x.endswith(".pth"), files))
        if len(possible_checkpoints) != 1:
            raise ValueError(
                f"Too many checkpoint files in the given checkpoint directory. Please specify the model you want to load directly."
            )
        ckpt_path = f'{ckpt_path}/{possible_checkpoints[0]}'
        ckpt_dict = torch.load(ckpt_path, weights_only=False, map_location=device)

    # ugh, this is so ugly, something something hindsight something something 20-20
    # FIXME: probably should do a pattern match, but this works for now
    kwargs = str(Path(ckpt_path).parent).split('/')[-1].split('__')

    # model_name, lr, normalize, drop_p = pickle.load(ckpt_path + ".cfg")

    # strip 'model_' from the name
    model_name = kwargs[1][6:]
    lr = float(kwargs[2].split('_')[-1])
    normalize = int(kwargs[3].split('_')[-1])
    dropout_p = float(kwargs[4].split('_')[-1])

    model = ckpt_dict['model'](model_name=model_name, dropout_p=dropout_p)
    model.load_state_dict(ckpt_dict['model_state_dict'])

    # FIXME: add optim class and args to state dict
    optim = ckpt_dict.get('optimizer', torch.optim.AdamW)(
        lr=lr,
        params=model.classifier.parameters()
    ).load_state_dict(ckpt_dict['optimizer_state_dict'])

    return {'model': model, 'optim': optim, 'normalize': normalize}
