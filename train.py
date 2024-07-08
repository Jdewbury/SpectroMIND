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
from dataset import Spectra30Class

import sys; sys.path.append("..")

def train_model(args):
    folder_name = f'{args.model}{"-" + args.activation if args.model == "resnet" else ""}/{args.train_time}-{args.seed}/{args.optimizer}{"-" + args.base_optimizer if args.optimizer in ["SAM", "ASAM"] else ""}'

    if args.save:
        count = 1
        unique_dir = f'results/{folder_name}_{count}'
        
        while os.path.exists(unique_dir):
            count += 1
            unique_dir = f'results/{folder_name}_{count}'
        
        os.makedirs(unique_dir, exist_ok=True)
        np.save(f'{unique_dir}/params.npy', vars(args))

    dataset = Spectra30Class(args.spectra_dir, args.label_dir, args.train_split, args.test_split, args.spectra_test_dir, args.label_test_dir, args.shuffle, args.seed, 2, args.batch_size)
    log = Log(log_each=10)

    if args.model == 'resnet':
        print('Using ResNet')
        hidden_sizes = [args.hidden_size] * args.layers
        num_blocks = [args.block_size] * args.layers
        model = ResNet(hidden_sizes, num_blocks, input_dim=args.input_dim,
                    in_channels=args.in_channels, n_classes=args.n_classes, activation=args.activation)
    elif args.model == 'mlp_flip':
        print('Using MLP_Flip')
        model = MLPMixer1D_flip(input_dim=args.input_dim, patch_size=args.patch_size, in_channels=args.in_channels, 
                                hidden_dim=args.hidden_size, depth=args.depth, token_dim=args.token_dim, 
                                channel_dim=args.channel_dim, output_dim=args.n_classes)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = smooth_crossentropy

    optimizer, base_optimizer = get_optimizer(model, args.optimizer, args.learning_rate, args.base_optimizer, 
                                              args.rho, args.weight_decay)
    
    scheduler_optimizer = optimizer.base_optimizer if args.optimizer in ['SAM', 'ASAM'] else optimizer
    scheduler = get_scheduler(args.scheduler, scheduler_optimizer, args.epochs, args.lr_step)

    print('Starting Training')
    
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    time_epochs = []
    best_acc = 0
    total_time = 0

    for epoch in range(args.train_time):
        start_time = time.time()
        model.train()
        log.train(len_dataset=len(dataset.train))
        batch_loss = []
        batch_acc = []

        for train_index, batch in enumerate(dataset.train):
            inputs, targets = (b.to(device) for b in batch)
            
            if args.optimizer not in ['SAM', 'ASAM']:
                predictions = model(inputs)
                targets = targets.to(torch.long)

                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
            else:
                enable_running_stats(model)
                predictions = model(inputs)
                targets = targets.to(torch.long)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                accuracy = correct.float().mean().item()
                loss_avg = loss.mean().item()

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy) 
                
                log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])
                scheduler.step()
        
        end_time = time.time()
        train_epoch_time = end_time - start_time
        total_time += train_epoch_time
        time_epochs.append(train_epoch_time)

        epoch_loss_avg = np.mean(batch_loss)
        epoch_accuracy_avg = np.mean(batch_acc)
        train_loss.append(epoch_loss_avg)
        train_accuracy.append(epoch_accuracy_avg)

        model.eval()
        log.eval(len_dataset=len(dataset.val))

        batch_loss = []
        batch_acc = []

        with torch.no_grad():
            for batch in dataset.val:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                targets = targets.to(torch.long)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                accuracy = correct.float().mean().item()
                
                loss_avg = loss.mean().item()

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy)

                log(model, loss.cpu(), correct.cpu())

        epoch_loss_avg = np.mean(batch_loss)
        epoch_accuracy_avg = np.mean(batch_acc)
        val_loss.append(epoch_loss_avg)
        val_accuracy.append(epoch_accuracy_avg)

        if epoch_accuracy_avg > best_acc:
            best_acc = epoch_accuracy_avg
            if args.save:
                torch.save(model.state_dict(), f'{unique_dir}/best_val.pth')

        log.flush()

        if total_time >= args.train_time:
            print(f'Training finished after {total_time} seconds')
            break
    
    if args.save:
        model.load_state_dict(torch.load(f'{unique_dir}/best_val.pth'))
    
    model.eval()

    batch_loss = []
    batch_acc = []

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
            loss_avg=loss.mean().item()

            batch_loss.append(loss_avg)
            batch_acc.append(accuracy)

    end_time = time.time()
    inference_time = end_time - start_time

    test_loss = np.mean(batch_loss)
    test_accuracy = np.mean(batch_acc)
    log.test(test_loss, test_accuracy)

    scores = {
        'train-time': time_epochs,
        'train-loss': train_loss,
        'train-acc': train_accuracy,
        'val-loss': val_loss,
        'val-acc': val_accuracy,
        'test-time': inference_time,
        'test-loss': test_loss,
        'test-acc': test_accuracy,
    }

    if args.save:
        print(f'Saving values at {unique_dir}')
        np.save(f'{unique_dir}/scores.npy', scores)

    return scores, unique_dir if args.save else None

# Remove the if __name__ == "__main__": block