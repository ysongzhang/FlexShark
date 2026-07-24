# FlexShark

FlexShark is a framework for maliciously secure mixed-bitwidth transformer inference.

## Building

FlexShark requires:

- **Compiler**: C++20 compatible compiler (e.g., GCC or Clang)
- **Build System**: CMake (>= 3.16)
- **Eigen3**
- **OpenMP**

Install dependencies:
- Ubuntu
```bash
sudo apt install libeigen3-dev
```

To compile Tuna, run the following command from the project root:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

After compilation, all executables will be generated under:

```bash
./build/
```


## Benchmarking

We provide two benchmarking scripts in the root of project that can reproduce the benchmarking results from the paper (specifically, Table 3 and 4).

Supports models such as:
- BERT-base
- GPT2-base

Includes both:
- Offline phase (preprocessing with the dealer)
- Online phase (secure inference between 2 parties)

Benchmarks are executed via provided scripts for:
- Local execution (single machine)
- Remote execution (two machines)

### Local Setup

In case you want to run both parties of FlexShark on a single machine, with each party using 16 threads, the following commands can be used to run benchmarks.

```bash
bash run-benchmarks-local.sh
```

### Two Machine Setup

If you have two machines connected through IP, lets say at `10.0.0.1` and `10.0.0.2`, run the following commands on each machine to run benchmarks:

```bash
bash run-benchmarks-remote.sh 0 10.0.0.2 # on 10.0.0.1
bash run-benchmarks-remote.sh 1 10.0.0.1 # on 10.0.0.2
```


## Accuracy

The accuracy pipeline evaluates the effectiveness of secure transformer inference in FlexShark by comparing outputs with plaintext execution. Unlike simple functional tests, FlexShark provides a full end-to-end evaluation workflow, covering data preparation, model conversion, secure inference, and metric computation.

**Key Features**
- **End-to-End Evaluation Pipeline**  
From model fine-tuning to MPC inference and metric reporting, the entire workflow is automated and reproducible.
- **Plaintext Baseline Verification**  
FlexShark evaluates the exact same inputs and weights in plaintext before MPC, providing a reliable ground truth for comparison.
- **Automated Fixed-Point Conversion**
Floating-point weights and inputs are automatically converted into fixed-point secret shares, enabling seamless integration with MPC backends.
- **Support for Multiple Tasks**  
    - Classification: QNLI, RTE
    - Regression: STS-B

**See detailed instructions in:** [Accuracy Guide](acc.md).