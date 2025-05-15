from huggingface_hub import HfApi, upload_folder
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Upload trained HuBERT model to Hugging Face Hub')
parser.add_argument('--username', type=str, default="pgilles", 
                    help='Your Hugging Face username')
parser.add_argument('--model_dir', type=str, default="./fine_tuned_vowel_model", 
                    help='Directory containing the fine-tuned HuBERT model')
parser.add_argument('--repo_name', type=str, default="vowel-classifier-hubert2", 
                    help='Name of the repository on Hugging Face')
args = parser.parse_args()

# Construct repo_id
repo_id = f"{args.username}/{args.repo_name}"

# Initialize API
api = HfApi()

# Create or get repo
try:
    api.create_repo(repo_id=repo_id, exist_ok=True)
    print(f"✅ Repository {repo_id} is ready")
    
    # Upload local folder contents
    upload_folder(
        repo_id=repo_id,
        folder_path=args.model_dir,  # local model folder path
        path_in_repo=".",  # upload everything inside this folder
        commit_message="Upload fine-tuned HuBERT vowel classifier"
    )
    
    print(f"✅ Successfully uploaded HuBERT model to https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"❌ Error uploading model: {str(e)}")