import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from data_utils import HAM10000Dataset
from peft import LoraConfig, get_peft_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="sd-ham10000-lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=4)

    # 1. Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 2. Setup LoRA using PEFT (The recommended modern way)
    # This avoids the LoRAAttnProcessor versioning issues
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # 3. Setup Dataset
    ds_helper = HAM10000Dataset()
    train_dataset = ds_helper.get_prompt_dataset(tokenizer, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Prepare with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # 4. Training Loop
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_train_epochs}", disable=not accelerator.is_local_main_process)
        
        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(accelerator.device)
            input_ids = batch["input_ids"].to(accelerator.device)

            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
        progress_bar.close()

    # 5. Save the trained LoRA
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Use PEFT's save_pretrained to save only LoRA weights
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(args.output_dir)
        print(f"\nTraining complete! LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()
