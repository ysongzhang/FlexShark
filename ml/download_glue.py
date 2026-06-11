#!/usr/bin/env python3
"""
GLUE Dataset Downloader and Preprocessor for MPC-based BERT Inference

This script downloads the GLUE benchmark datasets (QNLI, RTE, STS-B) and
prepares them in the format required by the bert_acc.cpp benchmark.

The datasets used are the same as in the Mosformer paper (CCS 2025):
- QNLI: Question-answering NLI (classification, accuracy metric)
- RTE: Recognizing Textual Entailment (classification, accuracy metric)
- STS-B: Semantic Textual Similarity Benchmark (regression, Pearson/Spearman)

Output format:
- bert_input.npz: Contains tokenized input_ids for inference
- bert_weights.npz: Contains model weights (requires pretrained BERT)
- labels.txt: Ground truth labels for accuracy evaluation
"""

import os
import sys
import argparse
import numpy as np

try:
    from datasets import load_dataset
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
except ImportError:
    print("Required packages not found. Please install:")
    print("  pip install datasets transformers torch")
    sys.exit(1)


GLUE_TASKS = {
    'qnli': {
        'name': 'qnli',
        'num_labels': 2,
        'text_fields': ('question', 'sentence'),
        'label_field': 'label',
        'metric': 'accuracy',
    },
    'rte': {
        'name': 'rte',
        'num_labels': 2,
        'text_fields': ('sentence1', 'sentence2'),
        'label_field': 'label',
        'metric': 'accuracy',
    },
    'stsb': {
        'name': 'stsb',
        'num_labels': 1,
        'text_fields': ('sentence1', 'sentence2'),
        'label_field': 'label',
        'metric': 'pearson_spearman',
    },
}

# Fine-tuned BERT models for each GLUE task (from textattack)
# Using bert-base-uncased without fine-tuning produces random classifier outputs
# because BertForSequenceClassification adds a randomly initialized head
FINETUNED_MODELS = {
    'qnli': 'textattack/bert-base-uncased-QNLI',
    'rte': 'textattack/bert-base-uncased-RTE',
    'stsb': 'textattack/bert-base-uncased-STS-B',
}


def download_glue_dataset(task_name: str, split: str = 'validation', cache_dir: str = None):
    """Download GLUE dataset from Hugging Face."""
    print(f"[INFO] Downloading GLUE/{task_name} ({split} split)...")
    dataset = load_dataset('glue', task_name, split=split, cache_dir=cache_dir)
    print(f"[INFO] Downloaded {len(dataset)} samples")
    return dataset


def tokenize_dataset(dataset, task_config: dict, tokenizer, model=None, max_length: int = 128):
    """Tokenize dataset using BERT tokenizer and optionally generate embeddings."""
    print(f"[INFO] Tokenizing with max_length={max_length}...")

    text_fields = task_config['text_fields']
    label_field = task_config['label_field']

    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_embeddings = []

    for sample in dataset:
        text_a = sample[text_fields[0]]
        text_b = sample[text_fields[1]] if len(text_fields) > 1 else None

        return_tensors = 'pt' if model else 'np'

        encoded = tokenizer(
            text_a,
            text_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )

        if model:
            with torch.no_grad():
                input_ids = encoded['input_ids']
                token_type_ids = encoded['token_type_ids']
                # Generate embeddings using the model's embedding layer
                embeddings = model.bert.embeddings(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
                # embeddings shape: [1, Seq, Hidden]
                all_embeddings.append(embeddings.numpy()[0])
            
            all_input_ids.append(input_ids.numpy()[0])
            all_attention_masks.append(encoded['attention_mask'].numpy()[0])
        else:
            all_input_ids.append(encoded['input_ids'][0])
            all_attention_masks.append(encoded['attention_mask'][0])
            
        all_labels.append(sample[label_field])

    result = {
        'input_ids': np.array(all_input_ids, dtype=np.int64),
        'attention_mask': np.array(all_attention_masks, dtype=np.int64),
        'labels': np.array(all_labels, dtype=np.float32),
    }
    
    if model:
        result['input'] = np.array(all_embeddings, dtype=np.float32)
        print(f"[INFO] Generated embeddings with shape {result['input'].shape}")

    return result


def extract_bert_weights(model_name: str = 'bert-base-uncased', num_layers: int = 12, num_labels: int = 2):
    """
    Extract weights from pretrained BERT model in the format expected by bert_acc.cpp.

    Expected weight names:
    - l{i}_c_attn_w: Combined QKV projection weights [n_embd, 3*n_embd]
    - l{i}_c_attn_b: Combined QKV projection biases [3*n_embd]
    - l{i}_c_proj_w: Output projection weights [n_embd, n_embd]
    - l{i}_c_proj_b: Output projection biases [n_embd]
    - l{i}_ffn_up_w: FFN up-projection weights [n_embd, intermediate_size]
    - l{i}_ffn_up_b: FFN up-projection biases [intermediate_size]
    - l{i}_ffn_down_w: FFN down-projection weights [intermediate_size, n_embd]
    - l{i}_ffn_down_b: FFN down-projection biases [n_embd]
    """
    print(f"[INFO] Loading pretrained model: {model_name} (num_labels={num_labels})")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    weights = {}
    bert = model.bert

    for i in range(num_layers):
        layer = bert.encoder.layer[i]

        # Self-attention weights
        # BERT uses separate Q, K, V projections - we combine them for the MPC format
        q_w = layer.attention.self.query.weight.data.numpy()  # [n_embd, n_embd]
        k_w = layer.attention.self.key.weight.data.numpy()
        v_w = layer.attention.self.value.weight.data.numpy()

        q_b = layer.attention.self.query.bias.data.numpy()  # [n_embd]
        k_b = layer.attention.self.key.bias.data.numpy()
        v_b = layer.attention.self.value.bias.data.numpy()

        # Combine Q, K, V into single projection (GPT-2 style)
        c_attn_w = np.concatenate([q_w, k_w, v_w], axis=0).T  # [n_embd, 3*n_embd]
        c_attn_b = np.concatenate([q_b, k_b, v_b])  # [3*n_embd]

        # Output projection
        c_proj_w = layer.attention.output.dense.weight.data.numpy().T  # [n_embd, n_embd]
        c_proj_b = layer.attention.output.dense.bias.data.numpy()

        # FFN
        ffn_up_w = layer.intermediate.dense.weight.data.numpy().T  # [n_embd, intermediate]
        ffn_up_b = layer.intermediate.dense.bias.data.numpy()
        ffn_down_w = layer.output.dense.weight.data.numpy().T  # [intermediate, n_embd]
        ffn_down_b = layer.output.dense.bias.data.numpy()

        # LayerNorm
        ln1_w = layer.attention.output.LayerNorm.weight.data.numpy()
        ln1_b = layer.attention.output.LayerNorm.bias.data.numpy()
        ln2_w = layer.output.LayerNorm.weight.data.numpy()
        ln2_b = layer.output.LayerNorm.bias.data.numpy()

        prefix = f'l{i}_'
        weights[prefix + 'c_attn_w'] = c_attn_w.astype(np.float32)
        weights[prefix + 'c_attn_b'] = c_attn_b.astype(np.float32)
        weights[prefix + 'c_proj_w'] = c_proj_w.astype(np.float32)
        weights[prefix + 'c_proj_b'] = c_proj_b.astype(np.float32)
        weights[prefix + 'ffn_up_w'] = ffn_up_w.astype(np.float32)
        weights[prefix + 'ffn_up_b'] = ffn_up_b.astype(np.float32)
        weights[prefix + 'ffn_down_w'] = ffn_down_w.astype(np.float32)
        weights[prefix + 'ffn_down_b'] = ffn_down_b.astype(np.float32)
        weights[prefix + 'ln1_w'] = ln1_w.astype(np.float32)
        weights[prefix + 'ln1_b'] = ln1_b.astype(np.float32)
        weights[prefix + 'ln2_w'] = ln2_w.astype(np.float32)
        weights[prefix + 'ln2_b'] = ln2_b.astype(np.float32)

    # Extract Classifier
    try:
        classifier_w = model.classifier.weight.data.numpy().T
        classifier_b = model.classifier.bias.data.numpy()
        weights['classifier_w'] = classifier_w.astype(np.float32)
        weights['classifier_b'] = classifier_b.astype(np.float32)
        print("[INFO] Extracted classifier weights")
    except AttributeError:
        print("[WARN] Could not find classifier weights")

    print(f"[INFO] Extracted weights for {num_layers} layers")
    return weights


def prepare_glue_for_mpc(
    task_name: str,
    output_dir: str,
    max_samples: int = None,
    max_length: int = 128,
    model_name: str = 'bert-base-uncased',
    download_weights: bool = True,
):
    """
    Prepare GLUE dataset for MPC inference benchmark.

    Args:
        task_name: GLUE task name (qnli, rte, stsb)
        output_dir: Directory to save preprocessed files
        max_samples: Maximum number of samples to process (None for all)
        max_length: Maximum sequence length
        model_name: Pretrained BERT model name
        download_weights: Whether to download and extract model weights
    """
    if task_name.lower() not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Choose from: {list(GLUE_TASKS.keys())}")

    task_config = GLUE_TASKS[task_name.lower()]
    os.makedirs(output_dir, exist_ok=True)

    # Use fine-tuned model when the default base model is specified
    if model_name == 'bert-base-uncased' and task_name.lower() in FINETUNED_MODELS:
        model_name = FINETUNED_MODELS[task_name.lower()]
        print(f"[INFO] Using fine-tuned model for {task_name}: {model_name}")

    # Download dataset
    dataset = download_glue_dataset(task_config['name'], split='validation')

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[INFO] Using {len(dataset)} samples")

    # Load model for embedding generation
    print(f"[INFO] Loading model for embedding generation: {model_name}")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=task_config['num_labels'])
    model.eval()

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(dataset, task_config, tokenizer, model=model, max_length=max_length)

    # Save input data
    input_file = os.path.join(output_dir, 'bert_input.npz')
    np.savez(input_file, **tokenized)
    print(f"[INFO] Saved input data to {input_file}")

    # Save labels separately for accuracy evaluation
    labels_file = os.path.join(output_dir, 'labels.txt')
    np.savetxt(labels_file, tokenized['labels'], fmt='%.6f')
    print(f"[INFO] Saved labels to {labels_file}")

    # Save task metadata
    meta_file = os.path.join(output_dir, 'task_info.txt')
    with open(meta_file, 'w') as f:
        f.write(f"task_name: {task_name}\n")
        f.write(f"num_samples: {len(dataset)}\n")
        f.write(f"max_length: {max_length}\n")
        f.write(f"num_labels: {task_config['num_labels']}\n")
        f.write(f"metric: {task_config['metric']}\n")
        f.write(f"model_name: {model_name}\n")
    print(f"[INFO] Saved task info to {meta_file}")

    # Download and save weights
    if download_weights:
        weights = extract_bert_weights(model_name, num_labels=task_config['num_labels'])
        weights_file = os.path.join(output_dir, 'bert_weights.npz')
        np.savez(weights_file, **weights)
        print(f"[INFO] Saved model weights to {weights_file}")

    print(f"\n[SUCCESS] Dataset prepared in {output_dir}")
    print(f"  - Input file: bert_input.npz")
    print(f"  - Labels file: labels.txt")
    print(f"  - Weights file: bert_weights.npz")
    print(f"\nTo run accuracy benchmark:")
    print(f"  cp {output_dir}/bert_input.npz ml/datasets/glue/{task_name}/")
    print(f"  cp {output_dir}/bert_weights.npz ml/datasets/glue/{task_name}/")


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess GLUE datasets for MPC BERT benchmark'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        default='qnli',
        choices=['qnli', 'rte', 'stsb', 'all'],
        help='GLUE task to download (default: qnli)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: ./glue_{task})'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=100,
        help='Maximum number of samples to process (default: 100, use -1 for all)'
    )
    parser.add_argument(
        '--max-length', '-l',
        type=int,
        default=128,
        help='Maximum sequence length (default: 128)'
    )
    parser.add_argument(
        '--model-name', '-m',
        type=str,
        default='bert-base-uncased',
        help='Pretrained BERT model name (default: bert-base-uncased)'
    )
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Skip downloading model weights'
    )

    args = parser.parse_args()

    tasks = ['qnli', 'rte', 'stsb'] if args.task == 'all' else [args.task]
    max_samples = None if args.max_samples == -1 else args.max_samples

    for task in tasks:
        output_dir = args.output_dir or f'./datasets/glue/{task}'
        prepare_glue_for_mpc(
            task_name=task,
            output_dir=output_dir,
            max_samples=max_samples,
            max_length=args.max_length,
            model_name=args.model_name,
            download_weights=not args.no_weights,
        )


if __name__ == '__main__':
    main()
