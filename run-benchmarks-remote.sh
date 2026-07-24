set -e

for benchmark in bert, gpt2; do
  OMP_NUM_THREADS=16 ./build/benchmark-$benchmark 2 &> tmp2.txt
  sizeA=$(stat -c "%s" server.dat | awk '{printf "%.3f", $1/1024/1024}')
  sizeB=$(stat -c "%s" client.dat | awk '{printf "%.3f", $1/1024/1024}')
  total_size=$(echo "$sizeA + $sizeB" | bc -l | awk '{printf "%.3f", $0}')
  {
    echo "==== File size summary ===="
    echo "server.dat: ${sizeA} MB"
    echo "client.dat: ${sizeB} MB"
    echo "Total: ${total_size} MB"
  } >> tmp2.txt
  OMP_NUM_THREADS=16 ./build/benchmark-$benchmark $@ &> tmp.txt

  echo "Benchmarking $benchmark"
  echo "======================="
  echo "The dealer"
  echo "======================="
  cat tmp2.txt
  echo "Party"
  echo "======================="
  cat tmp.txt
  echo ""
  echo ""
  rm tmp.txt tmp2.txt
done

# If preprocessing materials are already generated, you can run the benchmarks for party 0 and party 1 only.
# for benchmark in bert, gpt2; do
#   OMP_NUM_THREADS=16 ./build/benchmark-$benchmark $@ &> tmp.txt

#   echo "Benchmarking $benchmark"
#   echo "======================="
#   echo "Party"
#   echo "======================="
#   cat tmp.txt
#   echo "======================="
#   rm tmp.txt
# done

