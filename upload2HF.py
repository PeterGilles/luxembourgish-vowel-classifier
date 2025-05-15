from huggingface_hub import HfApi, upload_folder

# Replace this with your actual HF username
username = "pgilles"
repo_name = "vowel-classifier-hubert2"
repo_id = f"{username}/{repo_name}"

# Initialize API and create repo
api = HfApi()
api.create_repo(repo_id=repo_id, private=False)

# Upload local folder contents (your fine-tuned model)
upload_folder(
    repo_id=repo_id,
    folder_path="./fine_tuned_vowel_model2",  # local model folder path
    path_in_repo=".",  # upload everything inside this folder
    commit_message="Upload fine-tuned HuBERT vowel classifier"
)

print(f"✅ Successfully uploaded to https://huggingface.co/{repo_id}")
