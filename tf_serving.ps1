
$PORTNUMBER = jq -r '.portNumber' .\config.json
$MODELNAME = jq -r '.modelName' .\config.json
$CONTAINERNAME = jq -r '.containerName' .\config.json
$MODELFULLPATH = jq -r '.modelFullPath' .\config.json
docker run -p "${PORTNUMBER}:${PORTNUMBER}" --name "${containerName}" -v C:\\Repositories\\mlops_homework\\mnist_model:/models/mnist_model -e MODEL_NAME="${MODELNAME}" -t tensorflow/serving
