import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
import argparse
from tqdm.auto import tqdm
from PIL import Image

def batch_generate(
    lora_path="/home/jovyan/ece285/sd-ham10000-lora", 
    output_dir="/home/jovyan/ece285/synthetic_dataset", 
    num_images_per_class=500,
    width=600,
    height=448,
    guidance_scale=7.5,
    num_inference_steps=30,
    gpu_id=0,
    classes=None
):
    """
    Batch generate HAM10000 skin lesion images.
    
    Args:
        width: 600 (aligned with original dataset, multiple of 8)
        height: 448 (aligned with original dataset, multiple of 8)
        gpu_id: GPU ID to use (0 or 1)
        classes: Generate only for these specified classes, e.g., ["akiec", "bcc"]. None = all classes
    """
    
    device = f"cuda:{gpu_id}"
    
    # 1. Base Model
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print(f"[GPU {gpu_id}] Loading base model from {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # 2. Load LoRA Weights
    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    config_file = os.path.join(lora_path, "adapter_config.json")
    
    if os.path.exists(adapter_file) and os.path.exists(config_file):
        print(f"[GPU {gpu_id}] LoRA weights found, loading...")
        # Properly load using PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet = pipe.unet.to(device)
        print(f"[GPU {gpu_id}] LoRA weights loaded successfully!")
    else:
        print(f"Error: Weight files not found! Please check path: {lora_path}")
        return

    # 3. Define HAM10000 Classes
    class_mapping = {
        "akiec": "actinic keratosis",
        "bcc": "basal cell carcinoma",
        "bkl": "benign keratosis",
        "df": "dermatofibroma",
        "mel": "melanoma",
        "nv": "melanocytic nevus",
        "vasc": "vascular lesion"
    }
    
    if classes is not None:
        class_mapping = {k: v for k, v in class_mapping.items() if k in classes}

    os.makedirs(output_dir, exist_ok=True)
    total = len(class_mapping) * num_images_per_class
    print(f"\n[GPU {gpu_id}] Generating {len(class_mapping)} classes, {num_images_per_class} images per class, resolution {width}x{height}.")
    print(f"[GPU {gpu_id}] Save directory: {output_dir}\n")

    # 4. Batch Generation
    for class_id, label_name in class_mapping.items():
        class_folder = os.path.join(output_dir, class_id)
        os.makedirs(class_folder, exist_ok=True)
        
        # Check for resume generation
        existing = len([f for f in os.listdir(class_folder) if f.endswith(".png")])
        if existing >= num_images_per_class:
            print(f"[GPU {gpu_id}] Class {class_id} already has {existing} images, skipping.")
            continue
        
        prompt = f"a high-quality dermoscopic image of {label_name}, neutral skin tone, bright professional lighting, clear clinical details"
        negative_prompt = "purple background, pinkish tint, red tint, dark edges, vignette, blurry, distorted, human, face, hair, watermark, text"
        
        print(f"[GPU {gpu_id}] Generating class: {class_id} ... already has {existing}/{num_images_per_class}")
        
        for i in tqdm(range(existing, num_images_per_class), desc=f"[GPU{gpu_id}] {class_id}"):
            with torch.autocast(device.split(":")[0]): # Handle cuda:0 appropriately
                image = pipe(
                    prompt, 
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,   # Apply new width
                    height=height  # Apply new height
                ).images[0]
            
            save_path = os.path.join(class_folder, f"{class_id}_{i:04d}.png")
            image.save(save_path)

    print(f"\n[GPU {gpu_id}] Generation complete! Images saved in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAM10000 Batch Image Generation Script (600x448)")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "gpu0", "gpu1"],
                        help="Run mode: single=all on one card, gpu0=split 1 on two cards, gpu1=split 2 on two cards")
    parser.add_argument("--num_per_class", type=int, default=500)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--lora_path", type=str, default="/home/jovyan/ece285/sd-ham10000-lora")
    parser.add_argument("--output_dir", type=str, default="/home/jovyan/ece285/synthetic_dataset")
    args = parser.parse_args()
    
    GPU0_CLASSES = ["akiec", "bcc", "bkl", "df"]
    GPU1_CLASSES = ["mel", "nv", "vasc"]
    
    if args.mode == "single":
        batch_generate(
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_images_per_class=args.num_per_class,
            width=args.width,
            height=args.height,
            gpu_id=0,
            classes=None
        )
    elif args.mode == "gpu0":
        batch_generate(
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_images_per_class=args.num_per_class,
            width=args.width,
            height=args.height,
            gpu_id=0,
            classes=GPU0_CLASSES
        )
    elif args.mode == "gpu1":
        batch_generate(
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_images_per_class=args.num_per_class,
            width=args.width,
            height=args.height,
            gpu_id=1,
            classes=GPU1_CLASSES
        )

