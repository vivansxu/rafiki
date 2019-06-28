LOG_FILEPATH=$PWD/logs/stop.log
source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

title "Stopping any existing jobs..."
python ./scripts/stop_all_jobs.py

title "Dumping database..." 
bash ./scripts/save_db.sh

# If database dump previously failed, prompt whether to continue script
if [ $? -ne 0 ]
then
    read -p "Failed to dump database. Continue? (y/n) " ok
    if [ $ok = "n" ]
    then
        exit 1
    fi
fi

title "Stopping Rafiki's DB..."
docker rm -f $POSTGRES_HOST || echo "Failed to stop Rafiki's DB"

title "Stopping Rafiki's Cache..."
docker rm -f $REDIS_HOST || echo "Failed to stop Rafiki's Cache"

title "Stopping Rafiki's Admin..."
docker rm -f $ADMIN_HOST || echo "Failed to stop Rafiki's Admin"

title "Stopping Rafiki's Advisor..."
docker rm -f $ADVISOR_HOST || echo "Failed to stop Rafiki's Advisor"

title "Stopping Rafiki's Web Admin..."
docker rm -f $ADMIN_WEB_HOST || echo "Failed to stop Rafiki's Web Admin"

echo "You'll need to destroy your machine's Docker swarm manually"

