# Optional: fetch weights from Hugging Face Hub instead of committing big files.
# pip install huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

def fetch():
    if hf_hub_download is None:
        raise ImportError("Install huggingface_hub to download weights.")
    # Replace with your model repo + filename
    path = hf_hub_download(repo_id="your-username/your-model-repo", filename="weights.safetensors")
    return path

if __name__ == "__main__":
    print(fetch())