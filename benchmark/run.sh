ARCH=$1
SERVING=$2
SEC=$3
LOG_FILE=$4

model="mcls"

header="      Date & Time,  Trans,  Elap Time,  Data Trans,  Resp Time,  Trans Rate,  Throughput,  Concurrent,    OKAY,   Failed"
echo "ARCH=$ARCH  SERVING=$SERVING ==> MODEL=$model" >> "$LOG_FILE"
echo "$header" >> "$LOG_FILE"

for batch_size in 1 8 32 64 128; do
    echo "BatchSize=$batch_size" >> "$LOG_FILE"
    for concurrency in 10 50 100; do
        python3.8 run.py -s "$SERVING" -m "$model" -c "$concurrency" -b "$batch_size" -t "$SEC"  -l "$LOG_FILE"
    done
done
printf "\n" >> "$LOG_FILE"
