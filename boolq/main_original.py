"""
Fine-tune TinyStories-1M on BoolQ.

Important:
- We explicitly obtain token embeddings with model.transformer.wte(...)
- We explicitly feed them back through model(inputs_embeds=...)
- The script trains the whole model by default.
- Validation loss and accuracy are plotted during training.

Install:
    pip install torch transformers datasets matplotlib tqdm
"""

import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "roneneldan/TinyStories-1M"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5

MAX_LENGTH = 256

# Set to None to use the complete dataset.
# Useful for quickly testing the script.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ============================================================
# Load model and tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)

model.train()

print(f"Model hidden size: {model.config.hidden_size}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# GPT-2-style tokenizers sometimes don't have a pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# BoolQ
# ============================================================

dataset = load_dataset("google/boolq")

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

if MAX_TRAIN_SAMPLES is not None:
    train_dataset = train_dataset.select(
        range(min(MAX_TRAIN_SAMPLES, len(train_dataset)))
    )

if MAX_VAL_SAMPLES is not None:
    val_dataset = val_dataset.select(
        range(min(MAX_VAL_SAMPLES, len(val_dataset)))
    )

print(f"Training examples:   {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")


# ============================================================
# Formatting
# ============================================================

def make_prompt(example):
    """
    Turn a BoolQ example into a simple generative task.

    Example:

        Passage:
        ...

        Question:
        Is ...?

        Answer:
    """

    return (
        "Passage:\n"
        + example["passage"]
        + "\n\nQuestion:\n"
        + example["question"]
        + "\n\nAnswer:"
    )


def tokenize_example(example):
    """
    Tokenize prompt and answer separately.

    We keep them separate because we want to explicitly construct:

        prompt_embeddings + answer_embeddings
    """

    prompt = make_prompt(example)

    # Leading spaces are intentional for GPT-2-style tokenization.
    answer = " yes" if example["answer"] else " no"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
    )["input_ids"]

    # Add EOS to the answer.
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    # Keep room for the answer.
    max_prompt_length = MAX_LENGTH - len(answer_ids)

    if max_prompt_length <= 0:
        raise ValueError("MAX_LENGTH is too small.")

    prompt_ids = prompt_ids[:max_prompt_length]

    return {
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
    }


train_tokenized = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
)

val_tokenized = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names,
)


# ============================================================
# Collator
# ============================================================

def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    attention_masks = []

    for item in batch:
        prompt_ids = item["prompt_ids"]
        answer_ids = item["answer_ids"]

        # Séquence complète : prompt + réponse
        input_ids = prompt_ids + answer_ids

        # Causal LM :
        # le token à la position i sert à prédire le token i+1.
        #
        # On ne veut calculer la loss que sur les tokens de la réponse.
        labels = (
            [-100] * (len(prompt_ids) - 1)
            + answer_ids
            + [-100]
        )

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append([1] * len(input_ids))

    # Longueur maximale dans ce batch
    max_len = max(len(x) for x in input_ids_list)

    padded_input_ids = []
    padded_labels = []
    padded_attention_masks = []

    for input_ids, labels, attention_mask in zip(
        input_ids_list,
        labels_list,
        attention_masks
    ):
        padding = max_len - len(input_ids)

        padded_input_ids.append(
            input_ids + [tokenizer.pad_token_id] * padding
        )

        padded_labels.append(
            labels + [-100] * padding
        )

        padded_attention_masks.append(
            attention_mask + [0] * padding
        )

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long
        ),
        "labels": torch.tensor(
            padded_labels,
            dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            padded_attention_masks,
            dtype=torch.long
        ),
    }

train_loader = DataLoader(
    train_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# Explicit embedding forward pass
# ============================================================

def forward_with_embeddings(batch):

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    # --------------------------------------------------------
    # THIS IS THE IMPORTANT PART
    # --------------------------------------------------------
    #
    # Normally Transformers does:
    #
    #     input_ids
    #         ↓
    #     embedding layer
    #         ↓
    #     transformer
    #
    # Here we explicitly perform the first part ourselves.
    #

    inputs_embeds = model.transformer.wte(input_ids)

    #
    # inputs_embeds has shape:
    #
    #     [batch, sequence_length, 64]
    #
    # You can modify it here.
    #
    # For example:
    #
    # inputs_embeds[..., 0] = 0
    #
    # would completely remove embedding dimension 0.
    #
    # --------------------------------------------------------

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )

    return outputs


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for batch in val_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # Explicit embedding lookup.
        inputs_embeds = model.transformer.wte(input_ids)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        batch_size = input_ids.shape[0]

        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size

        # ----------------------------------------------------
        # Evaluate YES/NO using the logits at the position
        # predicting the first answer token.
        # ----------------------------------------------------

        logits = outputs.logits

        for i in range(batch_size):

            # Find the final real prompt token.
            prompt_len = (
                attention_mask[i]
                .sum()
                .item()
                - len(
                    val_tokenized[
                        0
                    ]["answer_ids"]
                )
            )

            # Instead of relying on the above approximation,
            # simply find the first non--100 answer label.
            answer_positions = torch.where(
                labels[i] != -100
            )[0]

            if len(answer_positions) == 0:
                continue

            first_answer_position = answer_positions[0]

            # Due to causal shifting, logits at position
            # first_answer_position - 1 predict the first
            # answer token.
            prediction_position = first_answer_position - 1

            if prediction_position < 0:
                continue

            logits_at_position = logits[
                i,
                prediction_position,
            ]

            yes_ids = tokenizer(
                " yes",
                add_special_tokens=False,
            )["input_ids"]

            no_ids = tokenizer(
                " no",
                add_special_tokens=False,
            )["input_ids"]

            yes_id = yes_ids[0]
            no_id = no_ids[0]

            predicted_yes = (
                logits_at_position[yes_id]
                > logits_at_position[no_id]
            )

            true_yes = (
                labels[i, first_answer_position].item()
                == yes_id
            )

            if predicted_yes.item() == true_yes:
                correct += 1

    avg_loss = total_loss / total_examples
    accuracy = correct / total_examples

    model.train()

    return avg_loss, accuracy


# ============================================================
# Training
# ============================================================

train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    examples_seen = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for batch in progress:

        optimizer.zero_grad()

        outputs = forward_with_embeddings(batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        batch_size = batch["input_ids"].shape[0]

        running_loss += loss.item() * batch_size
        examples_seen += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / examples_seen

    val_loss, val_accuracy = evaluate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print()
    print(f"Epoch {epoch + 1}")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss:   {val_loss:.4f}")
    print(f"  Val acc:    {val_accuracy * 100:.2f}%")

    # Save best checkpoint.
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model.save_pretrained("tinystories-boolq-best")
        tokenizer.save_pretrained("tinystories-boolq-best")

        print("  Saved best model.")


# ============================================================
# Plot
# ============================================================

epochs = range(1, EPOCHS + 1)

plt.figure()

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train loss",
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("TinyStories-1M fine-tuning on BoolQ")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)

plt.show()


plt.figure()

plt.plot(
    epochs,
    [x * 100 for x in val_accuracies],
    marker="o",
)

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("BoolQ validation accuracy")

plt.grid(True)

plt.tight_layout()
plt.savefig("validation_accuracy.png", dpi=150)

plt.show()
