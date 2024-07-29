import argparse
import torch
import numpy as np
import time
import os

from utils.smooth_cross_entropy import smooth_crossentropy
from utils.log import Log
from utils.initialize import get_optimizer, get_scheduler
from utils.bypass_bn import enable_running_stats, disable_running_stats

from model.resnet_1d import ResNet
from model.mlp_flip import MLPMixer1D_flip
from dataset import RamanSpectra

import sys; sys.path.append("..")

def train_model(args, progress_callback=None, stop_flag=None):
    dir = f"{args['optimizer']}{'_' + args['base_optimizer'] if args['optimizer'] in ['SAM', 'ASAM'] else ''}_{args['seed']}"

    unique_dir = None
    if args['save']:
        count = 1
        unique_dir = f'train/{dir}_{count}'

        while os.path.exists(unique_dir):
            count += 1
            unique_dir = f'train/{dir}_{count}'

        os.makedirs(unique_dir, exist_ok=True)
        np.save(f'{unique_dir}/params.npy', vars(args))

    dataset = RamanSpectra(args['spectra_dir'], args['label_dir'], args['spectra_interval'], args['seed'],
                           args['shuffle'], num_workers=2, batch_size=args['batch_size'])
    log = Log(log_each=10)

    if args['model'] == 'resnet':
        hidden_sizes = [args['hidden_size']] * args['layers']
        num_blocks = [args['block_size']] * args['layers']
        model = ResNet(hidden_sizes, num_blocks, input_dim=args['input_dim'],
                       in_channels=args['in_channels'], num_classes=args['num_classes'], activation=args['activation'])
    elif args['model'] == 'mlp_flip':
        model = MLPMixer1D_flip(in_channels=args['in_channels'], input_dim=args['input_dim'],
                                num_classes=args['num_classes'], depth=args['depth'],
                                token_dim=args['token_dim'], channel_dim=args['channel_dim'],
                                patch_size=args['patch_size'])
    else:
        raise ValueError(f"Unknown model type: {args['model']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = smooth_crossentropy

    optimizer, base_optimizer = get_optimizer(model, args['optimizer'], args['learning_rate'], args['base_optimizer'],
                                              args['rho'], args['weight_decay'])

    scheduler_optimizer = optimizer.base_optimizer if args['optimizer'] in ['SAM', 'ASAM'] else optimizer
    scheduler = get_scheduler(args['scheduler'], scheduler_optimizer, args['epochs'], args['lr_step'])

    print('Starting Training')

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    time_epochs = []
    best_acc = 0

    for epoch in range(args['epochs']):
        if stop_flag and stop_flag():
            print("Training stopped by user")
            break

        start_time = time.time()
        model.train()
        log.train(len_dataset=len(dataset.train))
        batch_loss = []
        batch_acc = []

        for train_index, batch in enumerate(dataset.train):
            if stop_flag and stop_flag():
                print("Training stopped by user")
                break
            inputs, targets = (b.to(device) for b in batch)
            inputs = inputs

            if args['optimizer'] not in ['SAM', 'ASAM']:
                predictions = model(inputs)
                targets = targets.to(torch.long)

                loss = criterion(predictions, targets, smoothing=args['label_smoothing'])

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
            else:
                enable_running_stats(model)
                predictions = model(inputs)
                targets = targets.to(torch.long)
                loss = criterion(predictions, targets, smoothing=args['label_smoothing'])
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                criterion(model(inputs), targets, smoothing=args['label_smoothing']).mean().backward()
                optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                accuracy = correct.float().mean().item()
                loss_avg = loss.mean().item()

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy)

                log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])

        epoch_loss_avg = np.mean(batch_loss)
        epoch_accuracy_avg = np.mean(batch_acc)
        train_loss.append(epoch_loss_avg)
        train_accuracy.append(epoch_accuracy_avg)

        if stop_flag and stop_flag():
            print("Training stopped by user")
            break

        scheduler.step()

        model.eval()
        log.eval(len_dataset=len(dataset.val))

        batch_loss = []
        batch_acc = []

        with torch.no_grad():
            for batch in dataset.val:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                targets = targets.to(torch.long)
                loss = criterion(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                accuracy = correct.float().mean().item()

                try:
                    loss_avg = loss.mean().item()
                except:
                    loss_avg = 0

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy)

                log(model, loss.cpu(), correct.cpu())

        val_epoch_loss_avg = np.mean(batch_loss)
        val_epoch_accuracy_avg = np.mean(batch_acc)
        val_loss.append(val_epoch_loss_avg)
        val_accuracy.append(val_epoch_accuracy_avg)

        end_time = time.time()
        epoch_time = end_time - start_time
        time_epochs.append(epoch_time)

        if progress_callback:
            progress_data = progress_callback(epoch + 1, args['epochs'], val_epoch_loss_avg, val_epoch_accuracy_avg)
            yield progress_data

        if val_epoch_accuracy_avg > best_acc and args['save']:
            best_acc = val_epoch_accuracy_avg
            torch.save(model.state_dict(), f'{unique_dir}/best_val.pth')

    log.done()

    scores = {
        'train-time': time_epochs,
        'train-loss': train_loss,
        'train-acc': train_accuracy,
        'val-loss': val_loss,
        'val-acc': val_accuracy,
    }

    if args['save']:
        print(f'Saving values at {unique_dir}')
        np.save(f'{unique_dir}/scores.npy', scores)

    return scores, unique_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'mlp_flip'], help='Model architecture to use.')
    parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs.')
    parser.add_argument('--spectra_dir', nargs='+', default=['data/spectral_data/X_2018clinical.npy', 'data/spectral_data/X_2019clinical.npy'], help='Directory to spectra.')
    parser.add_argument('--label_dir', nargs='+', default=['data/spectral_data/y_2018clinical.npy', 'data/spectral_data/y_2019clinical.npy'], help='Directory to labels.')
    parser.add_argument('--spectra_interval', nargs='+', type=int, default=[400, 100], help='Specified patient intervals for clinical significance.')
    parser.add_argument('--weight_dir', default='models', type=str, help='Directory containing model weight(s).')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for the training and validation loops.')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--in_channels', default=64, type=int, help='Number of input channels.')
    parser.add_argument('--num_classes', default=5, type=int, help='Number of classes in the classification task.')
    parser.add_argument('--input_dim', default=1000, type=int, help='Input dimension for the model.')
    parser.add_argument('--rho', default=0.05, type=float, help='Rho value for the optimizer.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for the optimizer.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay value for the optimizer.')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Use 0.0 for no label smoothing.')
    parser.add_argument('--optimizer', default='SAM', type=str, choices=['SAM','ASAM','Adam', 'SGD'], help='Optimizer to be used.')
    parser.add_argument('--base_optimizer', default='SGD', type=str, choices=['Adam', 'SGD'], help='Base optimizer to be used.')
    parser.add_argument('--scheduler', default='step', type=str, choices=['step', 'cosine'], help='Learning rate scheduler to be used.')
    parser.add_argument('--lr_step', default=0.2, type=float, help='Step size for learning rate scheduler.')
    parser.add_argument('--seed', default=42, type=int, help='Initialization seed.')
    parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training set.')
    parser.add_argument('--save', action='store_true', help='Save results.')
    parser.add_argument('--train_split', default=0.7, type=float, help='Proportion of data to use for training.')
    parser.add_argument('--test_split', default=0.15, type=float, help='Proportion of data to use for testing.')

    # ResNet specific arguments
    parser.add_argument('--layers', default=6, type=int, help='Number of layers in ResNet.')
    parser.add_argument('--hidden_size', default=100, type=int, help='Hidden size in ResNet.')
    parser.add_argument('--block_size', default=2, type=int, help='Block size in ResNet.')
    parser.add_argument('--activation', default='selu', type=str, choices=['relu', 'selu', 'gelu'], help='Activation function in ResNet.')

    # MLP Flip specific arguments
    parser.add_argument('--depth', default=2, type=int, help='Depth of MLP Flip.')
    parser.add_argument('--token_dim', default=64, type=int, help='Token dimension in MLP Flip.')
    parser.add_argument('--channel_dim', default=16, type=int, help='Channel dimension in MLP Flip.')
    parser.add_argument('--patch_size', default=50, type=int, help='Patch size in MLP Flip.')

    args = parser.parse_args()
    scores, save_dir = train_model(args)
    print(f"Training completed. Scores: {scores}")
    if save_dir:
        print(f"Results saved in: {save_dir}")
    else:
        print("Results were not saved (save option was not selected).")