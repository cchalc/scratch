flytectl demo start
export FLYTECTL_CONFIG=/Users/christopher.chalcraft/.flyte/config-sandbox.yaml
curl https://raw.githubusercontent.com/minio/docs/master/source/extra/examples/minio-dev.yaml -O
kubectl apply -f minio-dev.yaml
