# Accuracy Evaluation Workflow Guide

This document provides a comprehensive guide to the Accuracy Evaluation workflow in FlexShark framework. It explains the step-by-step process of preparing data, converting weights, running secure multi-party computation (MPC) inference, and analyzing the results.

---

## 1. Prerequisites & Dependencies

Before starting the accuracy evaluation workflow, ensure your environment is correctly configured:

### 1.1 Environment Requirements

- **OS**: Linux
- **Compiler**: C++20 compatible compiler (e.g., GCC or Clang)
- **Build System**: CMake (>= 3.16)
- **Python**: Python 3.8+

### 1.2 Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Key dependencies include `torch`, `transformers`, `datasets`, `numpy`, and `scipy`.

### 1.3 Compilation

You must build the C++ MPC executables before running the evaluation:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## 2. Workflow & Data Flow

The accuracy evaluation consists of three main phases: **Data Preparation**, **Plaintext Baseline Evaluation**, and **Secure MPC Inference**.

### Phase 1: Data Preparation

The first step is to download datasets, fine-tune models (if necessary), and extract weights and inputs into `.npz` format. All scripts for this phase reside in the `ml/` directory.

#### 1.1 BERT (GLUE Tasks: QNLI, RTE, STS-B)

For BERT models, we use a custom fine-tuning script (`ml/finetune_glue.py`) to generate the model checkpoints used for evaluation. Note that this fine-tuning step is only intended to emulate the model preparation process of the model owner and provide a complete usage example for readers. FlexShark itself does not require any model retraining.

```bash
# Fine-tune QNLI: Run 5 epochs with 3 different random seeds, preserving the best model. Extract up to 1000 validation samples.
python3 ml/finetune_glue.py --task qnli --epochs 5 --num-seeds 3 --max-samples 1000

# Fine-tune RTE: Run 10 epochs across 5 seeds. Stop early if validation doesn't improve for 5 epochs. Use a smaller batch size (8) and learning rate (1e-5) for this small dataset.
python3 ml/finetune_glue.py --task rte --epochs 10 --num-seeds 5 --patience 5 --batch-size 8 --lr 1e-5

# Fine-tune STS-B: Run 10 epochs with a larger batch size (64). Extract up to 1000 validation samples.
python3 ml/finetune_glue.py --task stsb --epochs 10 --batch-size 64 --max-samples 1000
```

**Key Arguments for `finetune_glue.py`:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--task` | `qnli` | The GLUE task to fine-tune (`qnli`, `rte`, `stsb`, or `all`). |
| `--epochs` | Task-specific | Number of training epochs. |
| `--batch-size` | Task-specific | Training batch size. |
| `--lr` | `2e-5` | Learning rate for the optimizer. |
| `--max-samples` | `-1` (All) | Maximum number of validation samples to export to the `.npz` file. |
| `--num-seeds` | `1` | Runs training multiple times with different seeds and saves the best model. Useful for unstable small datasets like RTE. |
| `--patience` | Disabled | Early stopping patience (number of epochs with no improvement before stopping). |
| `--freeze-encoder` | `False` | Freezes the BERT encoder and only trains the classifier head. Faster but may yield slightly lower accuracy. |

**Data Flow:**

- **Input**: HuggingFace GLUE dataset and `bert-base-uncased` model.
- **Output Location**: `ml/datasets/<task>/`
- **Output Files**:
  - `bert_weights.npz`: Flattened model weights.
  - `bert_input.npz`: Pre-embedded input states and attention masks.
  - `labels.txt`: Ground truth labels.



### Phase 2: Plaintext Baseline Evaluation

Before running the slow MPC inference, it is highly recommended to evaluate the exact same `.npz` inputs and weights in plaintext to establish a baseline.

```bash
# Run plaintext baseline for QNLI (using 1000 samples)
python3 scripts/calc_plaintext_baseline.py --task qnli --limit 1000

# Run plaintext baseline for RTE (using 277 samples)
python3 scripts/calc_plaintext_baseline.py --task rte --limit 277

# Run plaintext baseline for STS-B (using 1000 samples)
python3 scripts/calc_plaintext_baseline.py --task stsb --limit 1000

# Run plaintext baseline for GPT-2 (matching the 100 samples generated earlier)
python3 scripts/calc_plaintext_baseline.py --task gpt2 --limit 100
```

**Key Arguments for `calc_plaintext_baseline.py`:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--task`, `-t` | Required | Task name (`qnli`, `rte`, `stsb`, `gpt2`). |
| `--limit`, `-n` | All | Limits the evaluation to the first `N` samples. Ensure this matches the `max_samples` you will use in MPC. |
| `--batch-size`, `-b` | `32` | Inference batch size for calculating the baseline. |
| `--weights`, `-w` | Task-specific | Path to the `.npz` weights file. Automatically resolved if omitted. |
| `--input`, `-i` | Task-specific | Path to the `.npz` input file. Automatically resolved if omitted. |
| `--device`, `-d` | Auto | Computation device (`cpu`, `cuda`, `mps`). |

**Purpose**: Validates that the extracted `.npz` files produce the expected accuracy without the complexity of MPC.
**Data Flow:**

- **Input**: `ml/datasets/<task>/*.npz` and `labels.txt`.
- **Output Location**: `results/plaintext/`
- **Output File**: `<task>_plaintext_results.txt` (e.g., `qnli_plaintext_results.txt`).

### Phase 3: Secure MPC Inference

The final phase runs the actual secure 2-party computation using the C++ backend. This is orchestrated by `scripts/run_tests.py`.

```bash
# Make sure to run the preprocessing generation before conducting the accuracy test. If successful, server.dat and client.dat will be generated in the root directory.
OMP_NUM_THREADS=16 ./build/benchmark-bert-acc 2

# Run MPC accuracy test for QNLI
python3 scripts/run_tests.py --accuracy --task qnli --max-samples 1000 --threads 16

# Run MPC accuracy test for RTE
python3 scripts/run_tests.py --accuracy --task rte --max-samples 277 --threads 16

# Run MPC accuracy test for STS-B
python3 scripts/run_tests.py --accuracy --task stsb --max-samples 1000 --threads 16
```

**Key Arguments for `run_tests.py`:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--task` | All | Specifies the benchmark to run (e.g., `qnli`, `gpt2`, or `all`). |
| `--accuracy` | `False` | Triggers the accuracy evaluation mode (evaluates logits vs labels). Without this, it runs dummy performance benchmarks. |
| `--max-samples`, `-n` | All | Limits the evaluation size. Essential to prevent timeouts on slow MPC inference tasks. Must match the baseline `limit`. |
| `--threads`, `-j` | `os.cpu_count()` | Sets `OMP_NUM_THREADS` for C++ matrix operations. Set this to match your physical CPU cores. |

**Workflow inside `run_tests.py`:**

1. **Weight Conversion**: Automatically calls `scripts/convert_weights.py` to convert floating-point `.npz` files into fixed-point integer secret shares (`log/model_shares/` and `log/data_shares/`).
2. **Process Orchestration**: Spawns two C++ processes (Server/P0 and Client/P1). Server loads model shares; Client loads data shares.
3. **Execution**: The processes communicate via localhost to perform secure inference.
4. **Metric Parsing**: The script reads the raw output logits, compares them against `labels.txt`, and calculates the final metrics.

---

## 3. Directory Structure & Artifacts

During the workflow, several directories are used or created.

### 3.1 `ml/datasets/`

Stores the plaintext machine learning artifacts generated by Phase 1.

- `<task>/*_weights.npz`: Floating-point model weights.
- `<task>/*_input.npz`: Floating-point input embeddings.
- `<task>/labels.txt`: Ground truth labels used for scoring.

### 3.2 `log/` (Intermediate Directory)

Created during Phase 3 by the `convert_weights.py` script. It stores the fixed-point secret shares required by the C++ binaries.

- `log/model_shares/`: Contains `.npz` files of secret-shared weights (loaded by Server).
- `log/data_shares/`: Contains `.npz` files of secret-shared inputs (loaded by Client).

### 3.3 `results/`

The primary output directory for all evaluation results.

- `results/plaintext/`: Contains baseline results from `calc_plaintext_baseline.py`.
- `results/<model>_output_p0.txt`: Raw floating-point logits/predictions outputted by the MPC Server process (e.g., `bert_output_p0.txt`).
- `results/accuracy_summary.txt`: A formatted table summarizing the final metrics (Accuracy, Pearson/Spearman, or Perplexity) for the MPC runs.

---