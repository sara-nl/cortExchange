# Step by step for running a model

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
predictor.predict(input_path="/your/input/file.fits")
```