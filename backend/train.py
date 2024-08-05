import argparse
import torch
import numpy as np
import time
import os

from utils.smooth_cross_entropy import smooth_crossentropy
from utils.log import Log
from utils.initialize import get_optimizer, get_scheduler
from utils.bypass_bn import enable_running_stats, disable_running_stats

from dataset import RamanSpectra

import sys; sys.path.append("..")

from train_config import MODELS, MODEL_CONFIG, OPTIMIZERS, OPTIMIZER_CONFIG, GENERAL_CONFIG, SCHEDULER_CONFIG

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
        np.savez(f'{unique_dir}/params.npz', **args)

    dataset = RamanSpectra(args['spectra_dir'], args['label_dir'], args['spectra_interval'], args['seed'],
                           args['shuffle'], num_workers=2, batch_size=args['batch_size'])
    log = Log(log_each=10)

    # Use the MODELS dictionary to create the model
    model_creator = MODELS[args['model']]
    model = model_creator(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = smooth_crossentropy

    optimizer, base_optimizer = get_optimizer(model, args['optimizer'], args['learning_rate'], args['base_optimizer'],
                                              args.get('rho', 0.05), args['weight_decay'])

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
    
    # Add arguments based on GENERAL_CONFIG
    for key, config in GENERAL_CONFIG.items():
        parser.add_argument(f'--{key}', default=config['default'], type=eval(config['type']), help=f'{key} parameter')

    # Add model and optimizer arguments
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()), help='Model architecture to use.')
    parser.add_argument('--optimizer', type=str, choices=list(OPTIMIZERS.keys()), help='Optimizer to be used.')

    # Parse arguments
    args = parser.parse_args()

    # Convert args to dictionary
    args_dict = vars(args)

    # Add model-specific parameters
    if args.model in MODEL_CONFIG:
        for key, config in MODEL_CONFIG[args.model].items():
            if key not in args_dict:
                args_dict[key] = config['default']

    # Add optimizer-specific parameters
    if args.optimizer in OPTIMIZER_CONFIG:
        for key, config in OPTIMIZER_CONFIG[args.optimizer].items():
            if key not in args_dict:
                args_dict[key] = config['default']

    # Add scheduler parameters
    for key, config in SCHEDULER_CONFIG.items():
        if key not in args_dict:
            args_dict[key] = config['default']

    scores, save_dir = train_model(args_dict)
    print(f"Training completed. Scores: {scores}")
    if save_dir:
        print(f"Results saved in: {save_dir}")
    else:
        print("Results were not saved (save option was not selected).")