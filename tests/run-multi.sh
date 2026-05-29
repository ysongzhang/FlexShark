# set -e
# # get tname from argv
# tname=$1
# ./test-$tname 2 &> /dev/null
# ./test-$tname 1 &> /dev/null &
# ./test-$tname 0 &> /dev/null
# echo "passed"

set -e
# get tname from argv
tname=$1
OMP_NUM_THREADS=16 ./build/test-multi-$tname 2
OMP_NUM_THREADS=16 ./build/test-multi-$tname 1 &
OMP_NUM_THREADS=16 ./build/test-multi-$tname 0
echo "passed"

sizeA=$(stat -c "%s" server.dat | awk '{printf "%.3f", $1/1024/1024}')
sizeB=$(stat -c "%s" client.dat | awk '{printf "%.3f", $1/1024/1024}')
total_size=$(echo "$sizeA + $sizeB" | bc -l | awk '{printf "%.3f", $0}')
{
echo "==== File size summary ===="
echo "server.dat: ${sizeA} MB"
echo "client.dat: ${sizeB} MB"
echo "Total: ${total_size} MB"
}
