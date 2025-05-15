import os
import argparse
from huggingface_hub import HfApi, upload_folder

# Parse command line arguments
parser = argparse.ArgumentParser(description='Upload trained models to Hugging Face Hub')
parser.add_argument('--model_type', type=str, required=True, choices=['hubert', 'wav2vec2', 'both'], 
                    help='Model type to upload: hubert, wav2vec2, or both')
parser.add_argument('--username', type=str, default="pgilles", 
                    help='Your Hugging Face username')
parser.add_argument('--hubert_dir', type=str, default="./fine_tuned_vowel_model_hubert", 
                    help='Directory containing the fine-tuned HuBERT model')
parser.add_argument('--wav2vec2_dir', type=str, default="./fine_tuned_vowel_model_wav2vec2", 
                    help='Directory containing the fine-tuned Wav2Vec2 model')
args = parser.parse_args()

# Initialize API
api = HfApi()

def upload_model(model_type, model_dir):
    repo_name = f"vowel-classifier-{model_type}"
    repo_id = f"{args.username}/{repo_name}"
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Error: Directory {model_dir} does not exist!")
        return False
    
    # Create or get repo
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
        
        # Upload local folder contents
        upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,  # local model folder path
            path_in_repo=".",  # upload everything inside this folder
            commit_message=f"Upload fine-tuned {model_type.upper()} vowel classifier"
        )
        
        print(f"✅ Successfully uploaded {model_type} model to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error uploading {model_type} model: {str(e)}")
        return False

# Upload models based on selection
if args.model_type == 'hubert' or args.model_type == 'both':
    print(f"Uploading HuBERT model from {args.hubert_dir}...")
    upload_model('hubert', args.hubert_dir)

if args.model_type == 'wav2vec2' or args.model_type == 'both':
    print(f"Uploading Wav2Vec2 model from {args.wav2vec2_dir}...")
    upload_model('wav2vec2', args.wav2vec2_dir)

print("Upload process completed!")