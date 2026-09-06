import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

IMAGE_PATH = "data/outputs/candidate_patch_rgb.png"

QUESTION = (
    "Are there touching boundaries between any instance "
    "of complex cultivation patterns and urban fabrics?"
)


print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")


print("Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
)

model = model.to(device)


# --------------------------------------------------
# Create multimodal conversation
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
                "text": QUESTION,
            },
        ],
    }
]


# Create the prompt containing the image token
prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)


# IMPORTANT:
# Pass the actual PIL image here
inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt",
)


inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


print("\nRunning VQA...")
print("This may take a while on CPU.\n")


generated_ids = model.generate(
    **inputs,
    max_new_tokens=40,
)


generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]


print("=" * 60)

print("QUESTION:")
print(QUESTION)

print("\nMODEL OUTPUT:")
print(generated_text)

print("\nEXPECTED BENCHMARK ANSWER:")
print("yes")

print("=" * 60)