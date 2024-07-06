from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'npy'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload-multiple', methods=['POST'])
def upload_multiple_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files part'}), 400
        
        files = request.files.getlist('files[]')
        if len(files) == 0:
            return jsonify({'error': 'No selected files'}), 400
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
        
        return jsonify({'message': 'Files uploaded successfully'}), 200
    
    except Exception as e:
        print(f'Error: {str(e)}')  # Log the error to the console
        return jsonify({'error': str(e)}), 500

@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.npy')]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/file-data/<filename>', methods=['GET'])
def file_data(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            data = np.load(file_path)
            return jsonify({'data': data.tolist()})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-file', methods=['POST'])
def delete_file():
    data = request.get_json()
    filename = data.get('filename')

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'message': f'{filename} has been deleted'})
        else:
            return jsonify({'error': f'{filename} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

