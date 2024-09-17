# How to run a model

## Install through github automatically.

Install this package as follows:

```
pip install -e "git+https://github.com/sara-nl/cortExchange#egg=cortexchange"
```

You can then initialize the predictor as follows:

```Python
from cortexchange.predictor.surf.stop_predictor import StopPredictor

predictor = StopPredictor(device="cpu", model_name="name_of_the_model")
```

If you want to use a different remote host, or specify a new cache path, you can do so as follows:

The model cache will be used to store the to-be-downloaded models. This will then reuse downloaded models on subsequent
runs of the program.

```Python
from cortexchange.downloader import init_downloader

init_downloader(
    url="https://surfdrive.surf.nl/files/public.php/webdav/",
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

# Adding new models

## Implement Predictor class

We have a Predictor class which has the following two methods that need to be implemented:

```
def prepare_data(self, data: Any) -> torch.Tensor
def predict(self, data: torch.Tensor) -> Any
```

The prepare data ideally takes a path from which files are read, which would then automatically allow for seamless
integration with the cortexchange-cli tool.

There are also some optional methods that can be implemented:

```Python
def add_argparse_args(parser: argparse.ArgumentParser) -> None
def __init__(self, model_name: str, device: str, *args, your_additional_args=0, **kwargs)
```

Please ensure that your implementation of the `__init__` function can take an arbitrary amount of additional arguments.
We pass all the arguments parsed by the argparser to the `__init__` function.
Adding your model-specific argparser arguments should be done in the `add_argparse_args` method.