import requests
import json
from mlops.load_dataset import load_fashion_mnist_dataset

API_URL = "http://localhost:8501/v1/models/mnist_model:predict"


def make_prediction(instances):
    """Make a prediction"""
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(API_URL, data=data, headers=headers, timeout=60)
    predictions = json.loads(json_response.text)["predictions"]
    return predictions


if __name__ == "__main__":
    (_, _), (x_test, y_test) = load_fashion_mnist_dataset()
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_test = x_test.astype("float32") / 255.0
    predictions = make_prediction(x_test[0:4])
    print(predictions)
