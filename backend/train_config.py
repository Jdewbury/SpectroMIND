import torch.optim as optim
from model.resnet_1d import ResNet
from model.mlp_flip import MLPMixer1D_flip

def create_resnet(params):
    return ResNet(
        hidden_sizes=[params['hidden_size']] * params['layers'],
        num_blocks=[params['block_size']] * params['layers'],
        input_dim=params['input_dim'],
        in_channels=params['in_channels'],
        num_classes=params['num_classes'],
        activation=params['activation']
    )

def create_mlp_flip(params):
    return MLPMixer1D_flip(
        depth=params['depth'],
        token_dim=params['token_dim'],
        channel_dim=params['channel_dim'],
        input_dim=params['input_dim'],
        patch_size=params['patch_size'],
        num_classes=params['num_classes']
    )

MODELS = {
    'resnet': create_resnet,
    'mlp_flip': create_mlp_flip,
}

MODEL_CONFIG = {
    'resnet': {
        'layers': {'type': 'int', 'default': 6, 'min': 1, 'max': 50},
        'hidden_size': {'type': 'int', 'default': 100, 'min': 32, 'max': 1024},
        'block_size': {'type': 'int', 'default': 2, 'min': 1, 'max': 8},
        'activation': {'type': 'select', 'options': ['relu', 'selu', 'gelu']},
    },
    'mlp_flip': {
        'depth': {'type': 'int', 'default': 2, 'min': 1, 'max': 10},
        'token_dim': {'type': 'int', 'default': 64, 'min': 16, 'max': 256},
        'channel_dim': {'type': 'int', 'default': 16, 'min': 8, 'max': 128},
        'patch_size': {'type': 'int', 'default': 50, 'min': 10, 'max': 100},
    },
}

OPTIMIZERS = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
}

OPTIMIZER_CONFIG = {
    'Adam': {
        'weight_decay': {'type': 'float', 'default': 0.0005, 'min': 0, 'max': 0.1, 'step': 0.0001},
    },
    'SGD': {
        'momentum': {'type': 'float', 'default': 0.9, 'min': 0, 'max': 1, 'step': 0.1},
        'weight_decay': {'type': 'float', 'default': 0.0005, 'min': 0, 'max': 0.1, 'step': 0.0001},
    },
    'RMSprop': {
        'alpha': {'type': 'float', 'default': 0.99, 'min': 0, 'max': 1, 'step': 0.01},
        'weight_decay': {'type': 'float', 'default': 0.0005, 'min': 0, 'max': 0.1, 'step': 0.0001},
    },
}

GENERAL_CONFIG = {
    'epochs': {'type': 'int', 'default': 200, 'min': 1, 'max': 1000},
    'batch_size': {'type': 'int', 'default': 16, 'min': 1, 'max': 256},
    'learning_rate': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1, 'step': 0.0001},
    'in_channels': {'type': 'int', 'default': 64, 'min': 1, 'max': 256},
    'num_classes': {'type': 'int', 'default': 5, 'min': 2, 'max': 1000},
    'input_dim': {'type': 'int', 'default': 1000, 'min': 100, 'max': 10000},
    'label_smoothing': {'type': 'float', 'default': 0.1, 'min': 0, 'max': 1, 'step': 0.01},
    'seed': {'type': 'int', 'default': 42, 'min': 0, 'max': 9999},
    'shuffle': {'type': 'boolean', 'default': True},
    'save': {'type': 'boolean', 'default': False},
    'spectra_interval': {'type': 'text', 'default': '100'},
    'train_split': {'type': 'float', 'default': 0.7, 'min': 0.1, 'max': 0.9, 'step': 0.01},
    'test_split': {'type': 'float', 'default': 0.15, 'min': 0.1, 'max': 0.9, 'step': 0.01},
}

SCHEDULER_CONFIG = {
    'type': {'type': 'select', 'options': ['step', 'cosine'], 'default': 'step'},
    'step_size': {'type': 'int', 'default': 30, 'min': 1, 'max': 100},
    'gamma': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1, 'step': 0.01},
}