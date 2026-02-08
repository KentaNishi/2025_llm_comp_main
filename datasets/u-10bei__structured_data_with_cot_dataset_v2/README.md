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
    - name: estimated_tokens
      dtype: int64
  splits:
  - name: train
    num_bytes: 2731978
    num_examples: 2500
  download_size: 872806
  dataset_size: 2731978
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
