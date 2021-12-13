ARG TF_SERVING_VERSION=latest
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install schedule -i https://pypi.douban.com/simple

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install TF Serving pkg
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server
# COPY update.py /

EXPOSE 8500
EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
# RUN echo '#!/bin/bash \n\npython update.py &' >> /usr/bin/tf_serving_entrypoint.sh
RUN echo 'tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' >> /usr/bin/tf_serving_entrypoint.sh
RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
