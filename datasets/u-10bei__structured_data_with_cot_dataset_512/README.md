---
dataset_info:
  features:
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: metadata
    struct:
    - name: format
      dtype: string
    - name: complexity
      dtype: string
    - name: schema
      dtype: string
    - name: type
      dtype: string
    - name: estimated_tokens
      dtype: int64
  splits:
  - name: train
    num_bytes: 4208093
    num_examples: 3445
  download_size: 1307519
  dataset_size: 4208093
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
