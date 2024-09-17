# Step by step for running a model
## Install requirements
```shell
python -m venv venv
source activate venv/bin/activate
pip install -r requirements.txt
```

### Inference guide

Install this package as follows:
```
pip install -e "git+https://github.com/sara-nl/cortExchange"
```

You can then initialize the predictor as follows:

```Python
from cortexchange.predictor.stop_predictor import StopPredictor

predictor = StopPredictor(cache="/your/model/cache", device="cpu", model="name_of_the_model")
```

The model cache will be used to store the to-be-downloaded models. This will then reuse downloaded models on subsequent runs of the program.

```Python
predictor.predict(input_path="/your/input/file.fits")
```