#!/bin/bash

# LLM推論比較ツールを起動

echo "LLM推論比較ツールを起動しています..."
echo "ブラウザで http://localhost:5000 にアクセスしてください"
echo ""

cd "$(dirname "$0")"
python app.py
