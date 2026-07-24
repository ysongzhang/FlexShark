#!/usr/bin/env python3
"""
Fine-tune BERT for MPC Inference

Fine-tunes bert-base-uncased on GLUE tasks (QNLI, RTE, STS-B) with a classifier
that operates directly on the [CLS] hidden state.

Exports weights and inputs in the same format as download_glue.py for direct
compatibility with convert_weights.py and bert_acc.cpp.

Usage:
    python ml/finetune_glue.py --task qnli --batch-size 128 --epochs 5
    python ml/finetune_glue.py --task stsb --epochs 10 --batch-size 64
    python ml/finetune_glue.py --task rte --epochs 10 --num-seeds 5 --patience 5
"""

import os
import sys
import copy
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
    from datasets import load_dataset
except ImportError:
    print("Required packages not found. Please install:")
    print("  pip install datasets transformers torch")
    sys.exit(1)


# ============================================================================
# Task Configurations
# ============================================================================

GLUE_TASKS = {
    'qnli': {
        'name': 'qnli',
        'num_labels': 2,
        'text_fields': ('question', 'sentence'),
        'label_field': 'label',
        'metric': 'accuracy',
        'default_epochs': 3,
        'default_batch_size': 32,
    },
    'rte': {
        'name': 'rte',
        'num_labels': 2,
        'text_fields': ('sentence1', 'sentence2'),
        'label_field': 'label',
        'metric': 'accuracy',
        'default_epochs': 5,
        'default_batch_size': 16,
    },
    'stsb': {
        'name': 'stsb',
        'num_labels': 1,
        'text_fields': ('sentence1', 'sentence2'),
        'label_field': 'label',
        'metric': 'pearson_spearman',
        'default_epochs': 3,
        'default_batch_size': 32,
    },
}

# Fine-tuned models from textattack (used for --freeze-encoder mode)
FINETUNED_MODELS = {
    'qnli': 'textattack/bert-base-uncased-QNLI',
    'rte': 'textattack/bert-base-uncased-RTE',
    'stsb': 'textattack/bert-base-uncased-STS-B',
}


# ============================================================================
# Model Definition
# ============================================================================

class BertClassifierNoPooler(nn.Module):
    """
    BERT classifier that skips the pooler layer.

    Architecture: BertModel encoder -> [CLS] token -> dropout -> linear classifier
    This matches the MPC inference pipeline which does not implement the pooler
    (Dense(768,768) + Tanh) between the encoder and the classifier.
    """

    def __init__(self, num_labels: int, pretrained: str = 'bert-base-uncased', dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    @classmethod
    def from_finetuned(cls, task_name: str, num_labels: int, dropout: float = 0.1):
        """
        Create model by loading encoder weights from a fine-tuned textattack model.
        The encoder is already task-adapted; only the classifier head is new.
        Use with --freeze-encoder to only train the classifier.
        """
        finetuned_name = FINETUNED_MODELS[task_name]
        print(f"[INFO] Loading encoder from fine-tuned model: {finetuned_name}")

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        # Copy encoder weights (without pooler)
        model.bert = BertModel.from_pretrained(finetuned_name, add_pooling_layer=False)
        model.dropout = nn.Dropout(dropout)
        model.classifier = nn.Linear(model.bert.config.hidden_size, num_labels)
        model.num_labels = num_labels
        return model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Take [CLS] token (position 0) directly - no pooler
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ============================================================================
# Data Loading
# ============================================================================

def load_glue_data(task_name: str, split: str, tokenizer, max_length: int = 128):
    """Load and tokenize a GLUE dataset split."""
    config = GLUE_TASKS[task_name]
    dataset = load_dataset('glue', config['name'], split=split)
    text_fields = config['text_fields']
    label_field = config['label_field']

    def tokenize_fn(examples):
        texts_a = examples[text_fields[0]]
        texts_b = examples[text_fields[1]] if len(text_fields) > 1 else None
        return tokenizer(
            texts_a,
            texts_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column(label_field, 'labels')
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    return dataset


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_dataset, val_dataset, task_config, epochs, batch_size, lr, device,
                patience=None):
    """Fine-tune the model and return validation metrics.

    Tracks the best validation metric across epochs and restores the best
    checkpoint at the end of training. Optionally stops early if the metric
    hasn't improved for `patience` consecutive epochs.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    is_regression = task_config['num_labels'] == 1
    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    model.to(device)

    best_metric = -float('inf')
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)

            if is_regression:
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss = loss_fn(logits, labels.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        metrics = evaluate_model(model, val_loader, task_config, device)
        metric_str = format_metrics(metrics, task_config)

        # Track best checkpoint
        current = get_primary_metric(metrics, task_config)
        if current > best_metric:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            epochs_without_improvement = 0
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, {metric_str} ★ New best")
        else:
            epochs_without_improvement += 1
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, {metric_str}")

        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"  Early stopping: no improvement for {patience} epochs")
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint (metric={best_metric:.4f})")

    return best_metrics


def evaluate_model(model, dataloader, task_config, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels']

            logits = model(input_ids, attention_mask, token_type_ids)
            all_preds.append(logits.cpu())
            all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    if task_config['num_labels'] == 1:
        # Regression
        preds_np = preds.squeeze(-1).numpy()
        labels_np = labels.numpy()
        from scipy.stats import pearsonr, spearmanr
        pearson_corr, _ = pearsonr(preds_np, labels_np)
        spearman_corr, _ = spearmanr(preds_np, labels_np)
        return {'pearson': pearson_corr, 'spearman': spearman_corr}
    else:
        # Classification
        pred_classes = preds.argmax(dim=-1)
        correct = (pred_classes == labels).sum().item()
        total = len(labels)
        return {'accuracy': correct / total, 'correct': correct, 'total': total}


def format_metrics(metrics, task_config):
    """Format metrics as a string for logging."""
    if task_config['num_labels'] == 1:
        return f"pearson={metrics['pearson']:.4f}, spearman={metrics['spearman']:.4f}"
    else:
        return f"accuracy={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})"


def get_primary_metric(metrics, task_config):
    """Return the primary metric value for checkpoint comparison."""
    if task_config['num_labels'] == 1:
        return metrics['pearson']
    else:
        return metrics['accuracy']


# ============================================================================
# Weight Export (matches download_glue.py format exactly)
# ============================================================================

def export_weights(model, output_path, num_layers=12):
    """
    Export model weights in the same format as download_glue.py's extract_bert_weights().

    Weight keys per layer i:
    - l{i}_c_attn_w: concat([q_w, k_w, v_w], axis=0).T -> [768, 2304]
    - l{i}_c_attn_b: concat([q_b, k_b, v_b]) -> [2304]
    - l{i}_c_proj_w: weight.T -> [768, 768]
    - l{i}_c_proj_b: [768]
    - l{i}_ffn_up_w: weight.T -> [768, 3072]
    - l{i}_ffn_up_b: [3072]
    - l{i}_ffn_down_w: weight.T -> [3072, 768]
    - l{i}_ffn_down_b: [768]
    - l{i}_ln1_w/b: [768]
    - l{i}_ln2_w/b: [768]
    - classifier_w: weight.T -> [768, num_labels]
    - classifier_b: [num_labels]
    """
    weights = {}
    bert = model.bert

    for i in range(num_layers):
        layer = bert.encoder.layer[i]

        # Self-attention: separate Q, K, V -> combined
        q_w = layer.attention.self.query.weight.data.numpy()
        k_w = layer.attention.self.key.weight.data.numpy()
        v_w = layer.attention.self.value.weight.data.numpy()
        q_b = layer.attention.self.query.bias.data.numpy()
        k_b = layer.attention.self.key.bias.data.numpy()
        v_b = layer.attention.self.value.bias.data.numpy()

        c_attn_w = np.concatenate([q_w, k_w, v_w], axis=0).T
        c_attn_b = np.concatenate([q_b, k_b, v_b])

        # Output projection
        c_proj_w = layer.attention.output.dense.weight.data.numpy().T
        c_proj_b = layer.attention.output.dense.bias.data.numpy()

        # FFN
        ffn_up_w = layer.intermediate.dense.weight.data.numpy().T
        ffn_up_b = layer.intermediate.dense.bias.data.numpy()
        ffn_down_w = layer.output.dense.weight.data.numpy().T
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

    # Classifier head
    classifier_w = model.classifier.weight.data.numpy().T
    classifier_b = model.classifier.bias.data.numpy()
    weights['classifier_w'] = classifier_w.astype(np.float32)
    weights['classifier_b'] = classifier_b.astype(np.float32)

    np.savez(output_path, **weights)
    print(f"[INFO] Saved weights to {output_path}")
    print(f"  classifier_w shape: {classifier_w.shape}")
    print(f"  classifier_b shape: {classifier_b.shape}")


# ============================================================================
# Input Export (matches download_glue.py format exactly)
# ============================================================================

def export_inputs(model, tokenizer, task_config, output_path, max_samples=None, max_length=128):
    """
    Export inputs in the same format as download_glue.py's tokenize_dataset().

    Output keys:
    - input: [num_samples, 128, 768] float32 (embeddings)
    - attention_mask: [num_samples, 128] int64
    - input_ids: [num_samples, 128] int64
    - labels: [num_samples] float32
    """
    text_fields = task_config['text_fields']
    label_field = task_config['label_field']

    raw_dataset = load_dataset('glue', task_config['name'], split='validation')
    if max_samples:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

    model.eval()
    all_embeddings = []
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sample in raw_dataset:
        text_a = sample[text_fields[0]]
        text_b = sample[text_fields[1]] if len(text_fields) > 1 else None

        encoded = tokenizer(
            text_a,
            text_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            input_ids = encoded['input_ids']
            token_type_ids = encoded['token_type_ids']
            embeddings = model.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )
            all_embeddings.append(embeddings.numpy()[0])

        all_input_ids.append(input_ids.numpy()[0])
        all_attention_masks.append(encoded['attention_mask'].numpy()[0])
        all_labels.append(sample[label_field])

    result = {
        'input': np.array(all_embeddings, dtype=np.float32),
        'attention_mask': np.array(all_attention_masks, dtype=np.int64),
        'input_ids': np.array(all_input_ids, dtype=np.int64),
        'labels': np.array(all_labels, dtype=np.float32),
    }

    np.savez(output_path, **result)
    print(f"[INFO] Saved inputs to {output_path}")
    print(f"  embeddings shape: {result['input'].shape}")
    print(f"  num_samples: {len(all_labels)}")


# ============================================================================
# Main Pipeline
# ============================================================================

def finetune_and_export(task_name, output_dir, epochs=None, batch_size=None, lr=2e-5,
                        max_samples=None, max_length=128, device=None,
                        freeze_encoder=False, num_seeds=1, patience=None):
    """Fine-tune BERT without pooler on a GLUE task and export for MPC.

    When num_seeds > 1, runs training multiple times with different seeds
    (42, 43, ...) and keeps the model with the best validation metric.
    """
    task_config = GLUE_TASKS[task_name]
    epochs = epochs or task_config['default_epochs']
    batch_size = batch_size or task_config['default_batch_size']

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    mode = "freeze-encoder" if freeze_encoder else "full fine-tune"
    print(f"\n{'='*60}")
    print(f" Fine-tuning BERT (no pooler) on {task_name.upper()} [{mode}]")
    print(f" epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")
    if num_seeds > 1:
        print(f" multi-seed: {num_seeds} seeds (42..{42 + num_seeds - 1})")
    if patience is not None:
        print(f" early stopping: patience={patience}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    print("[INFO] Loading training data...")
    train_dataset = load_glue_data(task_name, 'train', tokenizer, max_length)
    print(f"[INFO] Training samples: {len(train_dataset)}")

    print("[INFO] Loading validation data...")
    val_dataset = load_glue_data(task_name, 'validation', tokenizer, max_length)
    print(f"[INFO] Validation samples: {len(val_dataset)}")

    # Multi-seed training loop
    overall_best_metric = -float('inf')
    overall_best_state = None
    overall_best_metrics = None

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if num_seeds > 1:
            print(f"\n--- Seed {seed} ({seed_idx + 1}/{num_seeds}) ---")

        # Create model
        if freeze_encoder:
            if seed_idx == 0:
                print(f"[INFO] Loading encoder from textattack fine-tuned model...")
            model = BertClassifierNoPooler.from_finetuned(task_name, num_labels=task_config['num_labels'])
            for param in model.bert.parameters():
                param.requires_grad = False
            if seed_idx == 0:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"[INFO] Trainable params: {trainable:,} / {total:,} (classifier only)")
            lr_used = max(lr, 1e-3)
            if seed_idx == 0:
                print(f"[INFO] Using lr={lr_used} for classifier-only training")
        else:
            if seed_idx == 0:
                print("[INFO] Creating BertClassifierNoPooler (bert-base-uncased)...")
            model = BertClassifierNoPooler(num_labels=task_config['num_labels'])
            lr_used = lr

        # Train
        print(f"[INFO] Starting fine-tuning...")
        metrics = train_model(model, train_dataset, val_dataset, task_config,
                              epochs, batch_size, lr_used, device, patience=patience)

        # Track best across seeds
        current = get_primary_metric(metrics, task_config)
        if num_seeds > 1:
            print(f"  Seed {seed} result: {format_metrics(metrics, task_config)}")

        if current > overall_best_metric:
            overall_best_metric = current
            model.cpu()
            overall_best_state = copy.deepcopy(model.state_dict())
            overall_best_metrics = metrics
            if num_seeds > 1:
                print(f"  ★ New overall best (seed {seed})")

    # Restore overall best model
    if freeze_encoder:
        model = BertClassifierNoPooler.from_finetuned(task_name, num_labels=task_config['num_labels'])
    else:
        model = BertClassifierNoPooler(num_labels=task_config['num_labels'])
    model.load_state_dict(overall_best_state)
    model.cpu()
    model.eval()
    metrics = overall_best_metrics

    # Print final metrics
    print(f"\n[RESULT] Final plaintext metrics for {task_name.upper()}:")
    print(f"  {format_metrics(metrics, task_config)}")

    # Export
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, 'bert_weights.npz')
    export_weights(model, weights_path)

    input_path = os.path.join(output_dir, 'bert_input.npz')
    export_inputs(model, tokenizer, task_config, input_path,
                  max_samples=max_samples, max_length=max_length)

    # Save labels
    raw_dataset = load_dataset('glue', task_config['name'], split='validation')
    if max_samples:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
    labels = np.array([s[task_config['label_field']] for s in raw_dataset], dtype=np.float32)
    labels_path = os.path.join(output_dir, 'labels.txt')
    np.savetxt(labels_path, labels, fmt='%.6f')
    print(f"[INFO] Saved labels to {labels_path}")

    # Save task metadata
    meta_path = os.path.join(output_dir, 'task_info.txt')
    with open(meta_path, 'w') as f:
        f.write(f"task_name: {task_name}\n")
        f.write(f"num_samples: {len(labels)}\n")
        f.write(f"max_length: {max_length}\n")
        f.write(f"num_labels: {task_config['num_labels']}\n")
        f.write(f"metric: {task_config['metric']}\n")
        f.write(f"model_name: bert-base-uncased (fine-tuned, no pooler)\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        if num_seeds > 1:
            f.write(f"num_seeds: {num_seeds}\n")
        if patience is not None:
            f.write(f"patience: {patience}\n")
    print(f"[INFO] Saved task info to {meta_path}")

    print(f"\n[SUCCESS] {task_name.upper()} fine-tuned and exported to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune BERT without pooler for MPC inference'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        default='qnli',
        choices=['qnli', 'rte', 'stsb', 'all'],
        help='GLUE task to fine-tune (default: qnli)',
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of training epochs (default: task-specific)',
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Training batch size (default: task-specific)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: ml/datasets/{task})',
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=-1,
        help='Max validation samples to export (default: -1 for all)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (default: auto-detect)',
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        help='Freeze encoder (from textattack model), only train classifier head. '
             'Much faster (~minutes on CPU) but may have slightly lower accuracy.',
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=1,
        help='Number of seeds to try (default: 1). Runs training with seeds 42, 43, ... '
             'and keeps the best model. Useful for small datasets like RTE.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='Early stopping patience (default: disabled). Stop training when validation '
             'metric has not improved for this many consecutive epochs.',
    )

    args = parser.parse_args()

    tasks = ['qnli', 'rte', 'stsb'] if args.task == 'all' else [args.task]
    max_samples = None if args.max_samples == -1 else args.max_samples

    all_metrics = {}
    for task in tasks:
        output_dir = args.output_dir or os.path.join('ml', 'datasets', task)
        metrics = finetune_and_export(
            task_name=task,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=max_samples,
            device=args.device,
            freeze_encoder=args.freeze_encoder,
            num_seeds=args.num_seeds,
            patience=args.patience,
        )
        all_metrics[task] = metrics

    # Summary
    print(f"\n{'='*60}")
    print(" Summary")
    print(f"{'='*60}")
    for task, metrics in all_metrics.items():
        task_config = GLUE_TASKS[task]
        print(f"  {task.upper()}: {format_metrics(metrics, task_config)}")


if __name__ == '__main__':
    main()
