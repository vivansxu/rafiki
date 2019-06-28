usage()  {
  echo "Usage: $0 <service_name> <rafiki_service_id> <rafiki_service_type> <worker_install_command> <replicas>"
  exit 1
}

if [ $# -ne 5 ] ; then
    usage
fi

docker service create --name $1 \
  --replicas $5 \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -e ADMIN_HOST=$ADMIN_HOST \
  -e ADMIN_PORT=$ADMIN_PORT \
  -e ADVISOR_HOST=$ADVISOR_HOST \
  -e ADVISOR_PORT=$ADVISOR_PORT \
  -e WORKER_INSTALL_COMMAND="$4" \
  -e RAFIKI_SERVICE_TYPE=$3 \
  -e RAFIKI_SERVICE_ID=$2 \
  -e WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
  -e DATA_DIR_PATH=$DATA_DIR_PATH \
  -e PARAMS_DIR_PATH=$PARAMS_DIR_PATH \
  -e LOGS_DIR_PATH=$LOGS_DIR_PATH \
  --mount type=bind,src=$HOST_WORKDIR_PATH/$DATA_DIR_PATH,dst=$DOCKER_WORKDIR_PATH/$DATA_DIR_PATH \
  --mount type=bind,src=$HOST_WORKDIR_PATH/$PARAMS_DIR_PATH,dst=$DOCKER_WORKDIR_PATH/$PARAMS_DIR_PATH \
  --mount type=bind,src=$HOST_WORKDIR_PATH/$LOGS_DIR_PATH:$DOCKER_WORKDIR_PATH,dst=$LOGS_DIR_PATH \
  $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION
