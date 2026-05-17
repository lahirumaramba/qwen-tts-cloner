import argparse
import os
import torch
import soundfile as sf
import librosa
from qwen_tts import Qwen3TTSModel

def clone_voice(ref_audio_path, ref_text, gen_text, output_path, model_id, device):
    print(f"Loading model: {model_id} on {device}...")
    
    # Set dtype based on device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    # Load the model
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype
    )
    
    print(f"Processing reference audio: {ref_audio_path}...")
    # 3-second voice cloning (In-Context Learning mode)
    # Using ref_text provides significantly better quality
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False if ref_text else True
    )
    
    print(f"Generating speech for text: \"{gen_text}\"...")
    # Generate the cloned voice audio
    wavs, sr = model.generate_voice_clone(
        text=gen_text,
        language="English",
        voice_clone_prompt=prompt_items
    )
    
    # FIX: Convert tensor to numpy array on CPU and normalize
    # wavs[0] is typically a tensor on the model's device
    audio_data = wavs[0]
    if hasattr(audio_data, "cpu"):
        audio_data = audio_data.cpu().numpy()
    
    # Normalize audio to prevent clipping and distortion
    import numpy as np
    if np.abs(audio_data).max() > 0:
        audio_data = audio_data / np.abs(audio_data).max() * 0.9
    
    # Save the output
    sf.write(output_path, audio_data, sr)
    print(f"Successfully saved cloned voice to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clone a voice using Qwen3-TTS")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio file (3-10s)")
    parser.add_argument("--ref_text", type=str, default="", help="Transcript of what is said in the reference audio")
    parser.add_argument("--ref_text_file", type=str, default="", help="Path to a text file containing the reference transcript")
    parser.add_argument("--gen_text", type=str, required=True, help="Text to generate in the cloned voice")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to save the generated audio")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Qwen TTS model ID")
    
    args = parser.parse_args()
    
    # Handle reference text from file
    ref_text = args.ref_text
    if args.ref_text_file:
        if os.path.exists(args.ref_text_file):
            with open(args.ref_text_file, "r") as f:
                ref_text = f.read().strip()
            print(f"Read reference transcript from file: {args.ref_text_file}")
        else:
            print(f"Warning: Reference text file not found at {args.ref_text_file}. Using --ref_text instead.")

    # Check for hardware acceleration
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    if not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio file not found at {args.ref_audio}")
        return

    clone_voice(
        args.ref_audio,
        ref_text,
        args.gen_text,
        args.output,
        args.model,
        device
    )

if __name__ == "__main__":
    main()
