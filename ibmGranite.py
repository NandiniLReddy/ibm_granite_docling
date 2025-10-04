"""Simple local script to run IBM Granite Docling model."""

import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

# Configuration - add another model id from huggingface model hub
MODEL_ID = "ibm-granite/granite-docling-258M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {DEVICE}...")

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map=DEVICE, 
    torch_dtype=torch.bfloat16
)

print("Model loaded successfully!")


def clean_response(text):
    """Remove special tokens from model output."""
    special_tokens = [
        "<|end_of_text|>", "<|end|>", "<|assistant|>", 
        "<|user|>", "<|system|>", "<pad>", "</s>", "<s>"
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    return text.strip()


def process_image(image_path, question):
    """Process an image with a question using the Granite model."""
    # Load and prepare image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # Apply chat template and process inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    
    # Decode and clean response
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )[0]
    
    return clean_response(generated_text)

if __name__ == "__main__":
    IMAGE_PATH = "/Users/nandinireddy/Desktop/IBM Granite/image-1.jpg"
    QUESTION = "what is the product sales revenue in year 2008"
    
    try:
        result = process_image(IMAGE_PATH, QUESTION)
        print("\n" + "="*50)
        print("RESULT:")
        print("="*50)
        print(result)
        
    except FileNotFoundError:
        print(f"Error: Image file '{IMAGE_PATH}' not found.")
        print("\nUsage:")
        print("1. Replace IMAGE_PATH with your actual image file path")
        print("2. Modify QUESTION as needed")
        print("\nExample questions:")
        print("- 'Convert this page to docling.'")
        print("- 'Extract all tables from this page.'")
        print("- 'Convert this table to OTSL.'")
        print("- 'Describe this image.'")