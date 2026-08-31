# How to run a model

## Install through github automatically.

Install this package as follows:

```
pip install -e "git+https://github.com/sara-nl/cortExchange#egg=cortexchange"
```

## Import and run the preferred model

You can then initialize the predictor as follows:

```Python
from cortexchange.architecture import get_architecture, Architecture

StopPredictor: type(Architecture) = get_architecture("surf/StopPredictor")
predictor = StopPredictor(device="cpu", model_name="name_of_the_model")
```

If you want to use a different remote host, or specify a new cache path, you can do so as follows:

The model cache will be used to store the to-be-downloaded models. This will then reuse downloaded models on subsequent
runs of the program.

```Python
from cortexchange.wdclient import init_downloader

init_downloader(
    url="https://researchdrive.surfsara.nl/public.php/webdav/",
    login="webdavlogin",
    password="password",
    cache="/your/new/cache/path/.cache"
) 
```

You can run prediction on the given model as follows:

```Python
data = predictor.prepare_data(input_path="/your/input/file.fits")
predictor.predict(data)
```

## TransferLearning architectures

`surf/TransferLearningV3` is the current version of the LOFAR calibrator-selection model;
`surf/TransferLearningV2` is kept for comparison. Both load checkpoints through the astroNNomy
package, so both apply the LoRA weights a checkpoint contains. They differ in how the input is
prepared, and therefore predict differently from the same weights:

| | V2 | V3 |
| --- | --- | --- |
| Normalization | inputs reach the model unnormalized: the mean/std recorded in the checkpoint were looked up in a way that always missed | the recorded mean/std are applied |
| Resampling | plain bilinear, with Gaussian noise added on every call, so predictions are stochastic | antialiased, and noise is opt-in through `--noise_sigma` |
| DINOv3 backbones | not supported | supported |

Both versions need the model package that the checkpoint was trained with, since a checkpoint
stores the model class itself:

```shell
pip install git+https://github.com/LOFAR-VLBI/astroNNomy.git#egg=astroNNomy
```

An astroNNomy older than the LoRA-loading fix is rejected with a message telling you to upgrade —
it would otherwise load these checkpoints with their adapters silently discarded.

A checkpoint with a DINOv3 backbone additionally needs a local DINOv3 checkout and Meta's
access-gated weights, since neither can be shipped:

```shell
export DINOV3_REPO_DIR=/path/to/facebookresearch/dinov3
export DINOV3_WEIGHTS=/path/to/dinov3_weights
```

DINOv2 checkpoints need neither.

# Adding new models

You can add new models to the code through the CLI if it uses existing architecture, or create a PR following the below
instructions.

## Implement Architecture class

We have a Architecture class which has the following two methods that need to be implemented:

```Python 
def prepare_data(self, data: Any) -> torch.Tensor:
    ...
    
    
def predict(self, data: torch.Tensor) -> Any:
    ...
```

The prepare data ideally takes a path from which files are read, which would then automatically allow for seamless
integration with the cortexchange-cli tool.

There are also some optional methods that can be implemented:

```Python
@staticmethod
def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    ...


def __init__(self, model_name: str, device: str, *args, your_additional_args=0, **kwargs):
    ...
```

Please ensure that your implementation of the `__init__` function can take an arbitrary amount of additional arguments.
We pass all the arguments parsed by the argparser to the `__init__` function.
Adding your model-specific argparser arguments should be done in the `add_argparse_args` method.

# cortExchange CLI

In the CLI we have a few tools: run, upload-weights, upload-architecture, list-group, and create-group.
Upload new models with an existing architecture easily with the `upload` tool.
List existing models in a group with `list-group`. `run` will download and run a single sample quickly.
This can be used to pre-download models, and see if they run on the hardware you have available.
The programmatical approach is better to use for integration with existing workflows, as then you can keep the model
weights loaded between inference passes through the model.
Uploading new architecture is possible temporarily through the CLI tool as well. This will zip the code and upload under
the given name. For longer term support, please create a PR with your model architecture, as this will add versioning
combined with the rest of the repository.

## Env variables

You can create a `.env` file where you are running containing the following two variables which are used for authentication to webdav:

```bash
WD_LOGIN=code
WD_PASSWORD=password
```

This will override the default read-only values. It has lower priority than values passed through the terminal.



# Acknowledgments
This repository is part of the project CORTEX (NWA.1160.18.316) of the research programme NWA-ORC which is (partly) financed by the Dutch Research Council (NWO). 
