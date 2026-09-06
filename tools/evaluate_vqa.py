import re
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

DATASET_PATH = "data/BigEarthNet.txt.parquet"

IMAGE_PATH = "data/outputs/candidate_patch_rgb.png"

# The BigEarthNet patch we downloaded
PATCH_NAME = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_32_64"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading benchmark dataset...")

df = pd.read_parquet(DATASET_PATH)

# Only benchmark samples
benchmark = df[df["split"] == "bench"].copy()

# Find questions belonging to our downloaded S2 patch
samples = benchmark[
    (benchmark["patch_id"].astype(str) == PATCH_NAME) &
    (benchmark["type"].astype(str).str.lower() == "binary")
].copy()

print("Questions found for this patch:", len(samples))


if len(samples) == 0:
    raise RuntimeError(
        "No benchmark questions found for the selected patch."
    )



# --------------------------------------------------
# Load image
# --------------------------------------------------

print("\nLoading image...")

image = Image.open(IMAGE_PATH).convert("RGB")


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)


# --------------------------------------------------
# Answer normalization
# --------------------------------------------------

def normalize_binary_answer(text):

    text = text.lower().strip()

    # Look for yes/no in the generated answer
    match = re.search(r"\b(yes|no)\b", text)

    if match:
        return match.group(1)

    return "unknown"


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

correct = 0
total = 0

yes_total = 0
yes_correct = 0

no_total = 0
no_correct = 0


print("\n" + "=" * 60)
print("STARTING VQA EVALUATION")
print("=" * 60)


for index, row in samples.iterrows():

    question = str(row["input"])
    expected = str(row["output"]).strip().lower()
    question_type = str(row["type"])

    print("\n" + "-" * 60)
    print("ID:", row["ID"])
    print("Type:", question_type)
    print("Category:", row["category"])
    print("Patch ID:", row["patch_id"])
    print("S1:", row["s1_name"])
    print("Expected:", expected)

    print("\nQuestion:")
    print(question)

    print("\nExpected:")
    print(expected)

    # --------------------------------------------------
    # Create multimodal prompt
    # --------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                        "Look carefully at the satellite image and answer the following "
                        "question using exactly one word.\n\n"
                        + question
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("\nRunning model...")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=30,
    )

    generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    )[0]

    # Only look at the assistant's response
    if "Assistant:" in generated_text:
        answer_text = generated_text.split("Assistant:")[-1].strip()
    else:
        answer_text = generated_text.strip()

    predicted = normalize_binary_answer(answer_text)

    is_correct = predicted == expected

    if expected == "yes":
        yes_total += 1
        if is_correct:
            yes_correct += 1

    elif expected == "no":
        no_total += 1
        if is_correct:
            no_correct += 1

    if is_correct:
        correct += 1

    total += 1

    print("\nRaw model output:")
    print(generated_text)

    print("\nNormalized prediction:")
    print(predicted)

    print("\nResult:")
    print("CORRECT ✅" if is_correct else "WRONG ❌")


# --------------------------------------------------
# Final result
# --------------------------------------------------

accuracy = correct / total if total else 0

yes_accuracy = yes_correct / yes_total if yes_total else 0
no_accuracy = no_correct / no_total if no_total else 0

print("\n")
print("=" * 60)
print("VQA BINARY BASELINE RESULTS")
print("=" * 60)

print(f"Patch: {PATCH_NAME}")
print(f"Total questions: {total}")

print(f"\nOverall:")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"\nYES questions:")
print(f"Correct: {yes_correct}/{yes_total}")
print(f"Accuracy: {yes_accuracy * 100:.2f}%")

print(f"\nNO questions:")
print(f"Correct: {no_correct}/{no_total}")
print(f"Accuracy: {no_accuracy * 100:.2f}%")

print("=" * 60)