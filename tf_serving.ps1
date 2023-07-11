
$PORTNUMBER = jq '.portNumber' .\config.json
$MODELNAME = jq '.modelName' .\config.json
$CONTAINERNAME = jq '.containerName' .\config.json
$MNISTPATH = "//C//Repositories//mlops_homework//mlops//mnist_model"
echo $MNISTPATH
echo $PORTNUMBER
echo $CONTAINERNAME
docker run -p 8501:8501 --name tfserving_fashion_mnist -v C:\\Repositories\\mlops_homework\\mnist_model:/models/mnist_model -e MODEL_NAME=mnist_model -t tensorflow/serving
