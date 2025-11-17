from huggingface_hub import HfApi

repo_id = "gtoscano/my-llama-qlora-imdb"   # exactly the name you want
folder_path = "/workspace/quantization/qlora-out"  # or ./qlora-out if you run from there

api = HfApi()

# Create the repo if it doesn't exist yet
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

# Upload all files in qlora-out/ to that repo
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model"
)

print(f"Uploaded {folder_path} to https://huggingface.co/{repo_id}")

