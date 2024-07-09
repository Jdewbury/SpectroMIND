from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os
import torch
from model.resnet_1d import ResNet
from model.mlp_flip import MLPMixer1D_flip
from flask_cors import CORS
import traceback

from model.resnet_1d import ResNet
from dataset import RamanSpectra
from utils.smooth_cross_entropy import smooth_crossentropy
import time
from sklearn import metrics

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'npy', 'pth'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 100 MB limit

executor = ThreadPoolExecutor()

@app.route('/')
def home():
    return "RamanML-Hub server is running!"

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
        print(f'Error: {str(e)}') 
        return jsonify({'error': str(e)}), 500

@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.npy')]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/file-data/<path:filename>', methods=['GET'])
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
            executor.submit(os.remove, file_path)
            return jsonify({'message': f'{filename} has been scheduled for deletion'})
        else:
            return jsonify({'error': f'{filename} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_directory_structure(path):
    structure = {}
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            structure[item] = get_directory_structure(item_path)
        elif item.endswith('.npy'):
            structure[item] = 'file'
    return structure

@app.route('/directory-structure', methods=['GET'])
def directory_structure():
    try:
        structure = get_directory_structure(app.config['UPLOAD_FOLDER'])
        return jsonify({'structure': structure})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-filtered-data', methods=['POST'])
def save_filtered_data():
    data = request.get_json()
    filename = data.get('filename')
    filtered_data = data.get('data')
    output_folder = data.get('outputFolder', '')  # Default to empty string if not provided

    if not filename or filtered_data is None:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        # Create a new filename for the filtered data
        base, ext = os.path.splitext(os.path.basename(filename))
        new_filename = f"{base}_filtered{ext}"
        
        # Create the output folder if it doesn't exist
        full_output_folder = os.path.join(app.config['UPLOAD_FOLDER'], output_folder)
        os.makedirs(full_output_folder, exist_ok=True)
        
        file_path = os.path.join(full_output_folder, new_filename)

        # Save the filtered data
        np.save(file_path, np.array(filtered_data))

        relative_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
        return jsonify({'message': f'Filtered data saved as {relative_path}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
"""
@app.route('/api/train', methods=['POST'])
def handle_train():
    try:
        print("Received training request")
        model_name = request.form.get('model')
        optimizer_name = request.form.get('optimizer')
        parameters = json.loads(request.form.get('parameters'))
        
        print(f"Model: {model_name}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Parameters: {parameters}")

        spectra_dirs = []
        print("Saving uploaded files...")
        for file in request.files.getlist('dataFolder'):
            filename = secure_filename(file.filename)
            spectra_dirs.append(filename)

        label_dirs = []
        for file in request.files.getlist('labelsFolder'):
            filename = secure_filename(file.filename)
            label_dirs.append(filename)

        print(label_dirs, spectra_dirs)
        # Prepare arguments for train_model function
        args = SimpleNamespace(
            model=model_name,
            spectra_dir=spectra_dirs,
            label_dir=label_dirs,
            optimizer=optimizer_name,
            train_time=100,  # Set default values or get from parameters
            batch_size=16,
            learning_rate=float(parameters.get('learning_rate', 0.001)),
            save=True,
            seed=42,
            shuffle=True,
            train_split=0.7,
            test_split=0.15,
            spectra_test_dir=None,
            label_test_dir=None,
            in_channels=1,
            n_classes=30,
            input_dim=1000,
        )

        # Add model-specific parameters
        if model_name == 'resnet':
            args.layers = int(parameters.get('layers', 6))
            args.hidden_size = int(parameters.get('hidden_size', 100))
            args.block_size = int(parameters.get('block_size', 2))
            args.activation = parameters.get('activation', 'relu')
        elif model_name == 'mlp_flip':
            args.depth = int(parameters.get('depth', 2))
            args.token_dim = int(parameters.get('token_dim', 64))
            args.channel_dim = int(parameters.get('channel_dim', 16))
            args.patch_size = int(parameters.get('patch_size', 50))

        # Add optimizer-specific parameters
        if optimizer_name in ['SAM', 'ASAM']:
            args.base_optimizer = parameters.get('base_optimizer', 'SGD')
            args.rho = float(parameters.get('rho', 0.05))
        
        args.momentum = float(parameters.get('momentum', 0.9))
        args.weight_decay = float(parameters.get('weight_decay', 0.0005))

        print("Starting training process...")
        # Run the training process
        scores, save_dir = train_model(args)

        print("Training completed successfully")
        return jsonify({
            'message': 'Training completed successfully',
            'scores': scores,
            'save_directory': save_dir
        }), 200

    except Exception as e:
        print(f"Error during training: {str(e)}")
        print(traceback.format_exc())  # This will print the full stack trace
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
"""

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        print("Received evaluation request")
        print("Files in request:", request.files)
        print("Form data:", request.form)
        
        for key, file in request.files.items():
            if file and not allowed_file(file.filename):
                print(f"Invalid file type: {file.filename}")
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
        
        if 'params' not in request.files:
            print("Missing params file")
            return jsonify({'error': 'Missing params file'}), 400

        if 'weights' not in request.files:
            print("Missing weights file")
            return jsonify({'error': 'Missing weights file'}), 400

        params_file = request.files['params']
        weights_file = request.files['weights']

        if not params_file.filename.endswith('.npy'):
            print("Invalid params file type")
            return jsonify({'error': 'Params file must be a .npy file'}), 400

        if not weights_file.filename.endswith('.pth'):
            print("Invalid weights file type")
            return jsonify({'error': 'Weights file must be a .pth file'}), 400

        dataset_files = [request.files[key] for key in request.files.keys() if key.startswith('dataset_')]
        label_files = [request.files[key] for key in request.files.keys() if key.startswith('label_')]

        if len(dataset_files) != len(label_files):
            print("Number of dataset and label files must match")
            return jsonify({'error': 'Number of dataset and label files must match'}), 400

        intervals = request.form.get('intervals')
        if not intervals:
            print("Missing spectra intervals")
            return jsonify({'error': 'Missing spectra intervals'}), 400

        intervals = [int(i) for i in intervals.split(',')]
        if len(intervals) != len(dataset_files):
            print("Number of intervals must match number of dataset files")
            return jsonify({'error': 'Number of intervals must match number of dataset files'}), 400

        if not all(allowed_file(f.filename) for f in [params_file, weights_file] + dataset_files + label_files):
            print("Invalid file type in uploaded files")
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded files
        params_path = os.path.join(UPLOAD_FOLDER, secure_filename(params_file.filename))
        weights_path = os.path.join(UPLOAD_FOLDER, secure_filename(weights_file.filename))
        dataset_paths = []
        label_paths = []

        print("Saving files...")
        params_file.save(params_path)
        weights_file.save(weights_path)

        for dataset_file, label_file in zip(dataset_files, label_files):
            dataset_path = os.path.join(UPLOAD_FOLDER, secure_filename(dataset_file.filename))
            label_path = os.path.join(UPLOAD_FOLDER, secure_filename(label_file.filename))
            dataset_file.save(dataset_path)
            label_file.save(label_path)
            dataset_paths.append(dataset_path)
            label_paths.append(label_path)

        # Load parameters and model weights
        print("Loading parameters and model weights...")
        params = np.load(params_path, allow_pickle=True).item()
        print(params)
        model = ResNet(
            hidden_sizes=[params['hidden_size']] * params['layers'],
            num_blocks=[params['block_size']] * params['layers'],
            input_dim=params['input_dim'],
            in_channels=params['in_channels'],
            num_classes=params['num_classes'],
            activation=params['activation']
        )
        
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        # Load dataset
        print("Loading dataset...")
        dataset = RamanSpectra(dataset_paths, label_paths, intervals, params['seed'], True,
                               num_workers=2, batch_size=params['batch_size'])

        # Perform evaluation
        print("Performing evaluation...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        batch_loss = []
        batch_acc = []
        all_predictions = []
        all_targets = []

        start_time = time.time()
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                inputs = inputs.float()

                predictions = model(inputs)
                targets = targets.to(torch.long)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                accuracy = correct.float().mean().item()
                loss_avg = loss.mean().item()

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy)

                all_predictions.extend(torch.argmax(predictions, 1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        end_time = time.time()
        inference_time = end_time - start_time

        test_loss = np.mean(batch_loss)
        test_accuracy = np.mean(batch_acc)

        # Calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(all_targets, all_predictions)
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_normalized_list = cm_normalized.tolist()
        class_labels_list = np.arange(params['num_classes']).tolist()

        print('Test Loss: ', test_loss, 'Test Acc: ', test_accuracy)

        results = {
            'test-time': inference_time,
            'test-loss': test_loss,
            'test-acc': test_accuracy,
            'confusion_matrix': cm_normalized_list,
            'class_labels': class_labels_list
        }

        print("Cleaning up uploaded files...")
        os.remove(params_path)
        os.remove(weights_path)
        for path in dataset_paths + label_paths:
            os.remove(path)

        print("Evaluation completed successfully")
        return jsonify({'results': results}), 200

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False)
