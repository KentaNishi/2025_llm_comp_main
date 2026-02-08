import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ファイルパス
INPUT_FILE = Path(__file__).parent.parent / "official_content" / "public_150.json"
OUTPUT_FILE = Path(__file__).parent.parent / "outputs" / "inference_base.json"


def load_data():
    """JSONファイルを読み込んでtask_idでマージ"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # task_idをキーとした辞書を作成
    output_dict = {item['task_id']: item.get('generation', '') for item in output_data}

    # 入力と出力をマージ
    merged_data = []
    for item in input_data:
        task_id = item['task_id']
        merged_data.append({
            'task_id': task_id,
            'task_name': item.get('task_name', ''),
            'query': item.get('query', ''),
            'output_type': item.get('output_type', ''),
            'generation': output_dict.get(task_id, '')
        })

    return merged_data


def save_output(task_id, generation):
    """出力ファイルを更新"""
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # 該当するtask_idを更新
    for item in output_data:
        if item['task_id'] == task_id:
            item['generation'] = generation
            break
    else:
        # 新規追加
        output_data.append({'task_id': task_id, 'generation': generation})

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/api/tasks')
def get_tasks():
    """全タスクのリストを取得"""
    data = load_data()
    return jsonify(data)


@app.route('/api/task/<task_id>')
def get_task(task_id):
    """特定のタスクを取得"""
    data = load_data()
    task = next((item for item in data if item['task_id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404


@app.route('/api/task/<task_id>', methods=['POST'])
def update_task(task_id):
    """タスクの出力を更新"""
    generation = request.json.get('generation', '')
    save_output(task_id, generation)
    return jsonify({'success': True, 'task_id': task_id})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
