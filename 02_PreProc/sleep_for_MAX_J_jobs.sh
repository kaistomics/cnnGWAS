#TEMPORAL LIMIT
MAX_J=40

while (true)
do
  NUM_J=`jobs -l| grep Running | wc -l`
  if [ $NUM_J -lt $MAX_J ]
  then
    break
  fi

  sleep 2
done
