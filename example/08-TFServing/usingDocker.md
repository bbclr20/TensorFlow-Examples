# Using TensorFlow Serving with Docker 

## Pulling a serving image

    docker pull tensorflow/serving

## Running a serving image

The serving images (both CPU and GPU) have the following properties:

- Port **8500** exposed for **gRPC**
- Port **8501** exposed for the **REST API**
- Optional environment variable MODEL_NAME (defaults to model)
- Optional environment variable MODEL_BASE_PATH (defaults to /models)

To serve with Docker:

    sudo docker run -p 8501:8501 \
    --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
    -e MODEL_NAME=my_model -t tensorflow/serving

When the serving image runs ModelServer, it runs it as follows:

    tensorflow_model_server --port=8500 --rest_api_port=8501 \
    --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}

Passing additional arguments:

    sudo docker run -p 8500:8500 8501:8501 \
    --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
    --mount type=bind,source=/path/to/my/models.config,target=/models/models.config \
    -t tensorflow/serving --model_config_file=/models/models.config

## Creating your own serving image

First run a serving image as a daemon:

    docker run -d --name serving_base tensorflow/serving

Next, copy your SavedModel to the container's model folder:

    docker cp models/<my model> serving_base:/models/<my model>

Finally, commit the container that's serving your model by changing MODEL_NAME to match your model's name `':

    docker commit --change "ENV MODEL_NAME <my model>" serving_base <my container>

## Serving example

First pull the serving image:

    docker pull tensorflow/serving

Clone the TensorFlow Serving repo which contains a model called Half Plus Two.

    mkdir -p /tmp/tfserving
    cd /tmp/tfserving
    git clone https://github.com/tensorflow/serving

Run the TensorFlow Serving container pointing it to this model and opening the REST API port (8501):

    sudo docker run -p 8501:8501 \
    --mount type=bind,\
    source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
    target=/models/half_plus_two \
    -e MODEL_NAME=half_plus_two -t tensorflow/serving &

sudo docker run -p 8501:8501 \
--mount type=bind,\
source=/home/ben/ProgTest/book/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
target=/models/half_plus_two \
-e MODEL_NAME=half_plus_two -t tensorflow/serving

To query the model using the predict API, you can run

    curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

This should return a set of values:

    { "predictions": [2.5, 3.0, 4.5] }
