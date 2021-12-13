# README


## Build

```bash
$ docker build -t <tag> .
```

## Convert

```bash
# tf to onnx
$ python3 -m tf2onnx.convert --saved-model models/tf-models-tfserving/mcls/100 --output models/ort-models-triton/mcls/100/model.onnx --opset 13

# onnx to tf
# onnx better use tensorflow==2.6.0, keras==2.6.0
# [python - Error importing tensorflow "AlreadyExistsError: Another metric with the same name already exists." - Stack Overflow](https://stackoverflow.com/questions/58012741/error-importing-tensorflow-alreadyexistserror-another-metric-with-the-same-nam)
$ onnx-tf convert -i /path/to/input.onnx -o /path/to/output
```

## Deploy

注意：[triton-inference-server/server](https://github.com/triton-inference-server/server) 可以推理各种框架的模型。

```bash
# tf-serving
$ docker run -it --rm \
-p 8501:8501 \
-p 8500:8500 \
-v "$(pwd)/models/tf-models-tfserving/:/models/" \
tensorflow/serving \
--model_config_file=/models/models.config \
--model_config_file_poll_wait_seconds=60

# triton + onnxruntime
$ docker run -it --rm \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
-v "$(pwd)/models/ort-models-triton:/models" \
nvcr.io/nvidia/tritonserver:21.11-py3 tritonserver \
--model-repository=/models

# triton + tf_saved_model
$ docker run -it --rm \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
-v "$(pwd)/models/tf-models-triton:/models" \
nvcr.io/nvidia/tritonserver:21.11-py3 tritonserver \
--model-repository=/models
```

文件目录：

```bash
models
├── ort-models-triton
│   └── mcls
│       ├── 100
│       │   └── model.onnx
│       └── config.pbtxt
├── tf-models-tfserving
│   ├── mcls
│   │   └── 100
│   │       ├── assets
│   │       ├── saved_model.pb
│   │       └── variables
│   │           ├── variables.data-00000-of-00001
│   │           └── variables.index
│   └── models.config
└── tf-models-triton
    └── mcls
        ├── 100
        │   └── model.savedmodel
        │       ├── assets
        │       ├── saved_model.pb
        │       └── variables
        │           ├── variables.data-00000-of-00001
        │           └── variables.index
        └── config.pbtxt
```

## Test

```bash
# 需启动 tf-serving 和 triton 任意一个服务
$ pytest test_server.py
```
