# LLM推論比較ツール

## 概要
このツールは、LLMの推論入力と出力を比較し、出力を編集できる軽量なWebアプリケーションです。

## 機能
- タスクIDをもとに入力（query）と出力（generation）を並べて表示
- 出力内容をブラウザ上で編集可能
- 編集内容をJSONファイルに保存
- HTMXを使用した高速なインタラクティブUI

## 必要な依存関係
```bash
pip install flask
```

## 起動方法
```bash
cd src
python app.py
```

ブラウザで `http://localhost:5000` にアクセスしてください。

## ファイル構造
```
src/
├── app.py              # Flaskアプリケーション
├── templates/
│   └── index.html      # メインUI（HTMX使用）
└── README.md           # このファイル
```

## 使用技術
- **バックエンド**: Flask (Python)
- **フロントエンド**: HTMX + Tailwind CSS
- **データ**: JSON形式
