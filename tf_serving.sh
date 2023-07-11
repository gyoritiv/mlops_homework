docker pull tensorflow/serving
docker run -p 8501:8501 --name tfserving_fashion_mnist
--mount type=bind,source=mnist_model,target=/models/mnist_model
-e MODEL_NAME=mnist_model -t tensorflow/serving
