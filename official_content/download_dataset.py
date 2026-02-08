import os
from huggingface_hub import snapshot_download

dataset_links = [
    "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset_512_v2",
    "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset_512_v4",
    "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset_512_v5",
    "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset_512",
    "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset_v2",
    "https://huggingface.co/datasets/u-10bei/structured_data_with_cot_dataset",
    "https://huggingface.co/datasets/daichira/structured-3k-mix-sft",
    "https://huggingface.co/datasets/daichira/structured-5k-mix-sft",
    "https://huggingface.co/datasets/daichira/structured-hard-sft-4k"
]

output_dir = "/home/nkutm/workspace/2025-llm-advance-competition-main/datasets"
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading datasets to: {output_dir}")

for link in dataset_links:
    # Extract dataset ID from the link
    dataset_id = link.replace("https://huggingface.co/datasets/", "")
    local_dataset_path = os.path.join(output_dir, dataset_id.replace('/', '__'))
    print(f"Downloading {dataset_id} to {local_dataset_path}")
    snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=local_dataset_path, local_dir_use_symlinks=False)

print("All specified datasets have been downloaded.")