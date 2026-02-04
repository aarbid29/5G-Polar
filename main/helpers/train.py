import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import time
import os
from tqdm import tqdm
from argparse import ArgumentParser
import json

from main.helpers.config import BiMambaConfig
from polar_bimamba_model import PolarBiMambaDecoder, SimplePolarBiMambaDecoder
from polar_bimamba_dataset import BER, FER, bin_to_sign
from polar_bimamba_init import initialize, save_checkpoint, dump_config


def train_epoch(model, device, train_loader, optimizer, epoch, lr, config: BiMambaConfig):
    """Train for one epoch"""
    model.train()
    cum_loss = cum_ber = cum_samples = 0.0
    t = time.time()
    
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}")
    ):
        z_mul = y * bin_to_sign(x)  # ground truth
        
        # Forward pass
        z_pred = model(magnitude.to(device), syndrome.to(device))
        
        # Compute loss
        loss, x_pred = model.loss(z_pred, z_mul.to(device), y.to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clipping)
        optimizer.step()
        
        # Metrics
        ber = BER(x_pred, x.to(device))
        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_samples += x.shape[0]
        
        if batch_idx == len(train_loader) - 1:
            logging.info(
                f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: '
                f'LR={lr:.2e}, Loss={cum_loss / cum_samples:.2e}, '
                f'BER={cum_ber / cum_samples:.2e}'
            )
    
    logging.info(f'Epoch {epoch} Train Time {time.time() - t:.2f}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples


def test(model, device, test_loader_list, EbNo_range_test):
    """Test model on multiple SNR values"""
    model.eval()
    results = {}
    total_ber = 0
    
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = cum_count = 0.0
            
            for m, x, z, y, magnitude, syndrome in tqdm(
                test_loader, desc=f"Test EbN0={EbNo_range_test[ii]}"
            ):
                z_pred = model(magnitude.to(device), syndrome.to(device))
                x_pred = model.get_codeword(z_pred, y.to(device))
                
                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
            
            test_ber /= cum_count
            ln_ber = -np.log(test_ber) if test_ber > 0 else float('inf')
            
            logging.info(
                f'Test EbN0={EbNo_range_test[ii]}, '
                f'BER={test_ber:.2e}, -ln(BER)={ln_ber:.2e}'
            )
            
            results[f"BER_{EbNo_range_test[ii]}"] = test_ber
            total_ber += test_ber / len(test_loader_list)
    
    results['test_ber'] = total_ber
    return results


def update_training_state(training_state, epoch, loss, ber):
    """Update training state dictionary"""
    training_state['epoch'] = epoch
    training_state['loss'] = loss
    training_state['BER'] = ber
    
    if ber < training_state.get('best_ber', float('inf')):
        training_state['best_ber'] = ber
        training_state['best_ber_epoch'] = epoch
    
    if loss < training_state.get('best_loss', float('inf')):
        training_state['best_loss'] = loss
        training_state['best_loss_epoch'] = epoch
    
    return training_state


def update_test_state(test_state, results, epoch):
    """Update test state with new results"""
    test_state.update(results)
    
    for key in results:
        best_key = f'best_{key}'
        best_result = test_state.get(best_key, float('inf'))
        
        if best_result <= results[key]:
            continue
        
        test_state[best_key] = results[key]
        test_state[f'{best_key}_epoch'] = epoch
    
    return test_state


def epoch_callback(
    config,
    training_state,
    model,
    optimizer,
    summary_writer,
    **kwargs
):
    """Callback after each epoch"""
    checkpoint = {
        'config': config,
        'state': training_state,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }
    
    if training_state['best_loss'] <= training_state['loss']:
        checkpoint['best_model'] = model.state_dict()
    
    save_checkpoint(checkpoint)
    
    # TensorBoard logging
    summary_writer.add_scalar('Train: Loss/Epoch', training_state['loss'], training_state['epoch'])
    summary_writer.add_scalar('Train: BER/Epoch', training_state['BER'], training_state['epoch'])
    summary_writer.add_scalar('Train: Best Loss/Epoch', training_state['best_loss'], training_state['epoch'])
    summary_writer.add_scalar('Train: Best BER/Epoch', training_state['best_ber'], training_state['epoch'])
    
    if (scheduler := kwargs.get('scheduler')):
        summary_writer.add_scalar('Train: LR/Epoch', scheduler.get_last_lr()[0], training_state['epoch'])


def test_callback(
    config,
    training_state,
    test_results,
    summary_writer,
    model,
    **kwargs
):
    """Callback after testing"""
    run_name = os.path.basename(os.path.normpath(config.path))
    hparams_dir = os.path.join(config.path, run_name)
    
    if os.path.exists(hparams_dir):
        for filename in os.listdir(hparams_dir):
            os.remove(os.path.join(hparams_dir, filename))
    
    # Log test results
    for key, value in test_results.items():
        if 'BER' in key and not key.endswith('epoch'):
            ln_ber = -np.log(value) if value > 0 else float('inf')
            summary_writer.add_scalar(f'Test: -ln({key})/Epoch', ln_ber, training_state['epoch'])
        else:
            summary_writer.add_scalar(f'Test: {key}/Epoch', value, training_state['epoch'])
        
        # Save best model for each metric
        if (best_result := test_results.get(f'best_{key}')) is not None and best_result >= value:
            torch.save(model.state_dict(), os.path.join(config.path, f'best_model_{key}'))
    
    # Hyperparameters
    summary_writer.add_hparams(
        dump_config(config),
        {**training_state, **test_results},
        run_name=run_name,
        global_step=training_state['epoch']
    )


def train_model(
    config,
    model,
    optimizer,
    training_state,
    dataset,
    summary_writer,
    epochs_per_test=10,
    scheduler_init=torch.optim.lr_scheduler.CosineAnnealingLR
):
    """Main training loop"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataloader, test_dataloader_list, _, EbNo_range_test = dataset
    
    epoch = training_state.get('epoch', 0)
    test_state = {}
    scheduler = None
    lr = config.warmup_lr
    
    for epoch in range(epoch + 1, config.epochs + 1):
        # Initialize scheduler after warmup
        if epoch >= config.warmup_length and scheduler is None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr
                param_group['initial_lr'] = param_group['lr']
            
            scheduler = scheduler_init(
                optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min,
                last_epoch=epoch - config.warmup_length
            )
        
        if epoch >= config.warmup_length and scheduler is not None:
            lr = scheduler.get_last_lr()[0]
        
        # Train epoch
        loss, ber = train_epoch(model, device, train_dataloader, optimizer, epoch, lr, config)
        update_training_state(training_state, epoch, loss, ber)
        
        if scheduler is not None:
            scheduler.step()
        
        # Epoch callback
        epoch_callback(
            config, training_state,
            model=model, optimizer=optimizer, scheduler=scheduler,
            summary_writer=summary_writer
        )
        
        # Test periodically
        if epoch % epochs_per_test == 0:
            results = test(model, device, test_dataloader_list, EbNo_range_test)
            update_test_state(test_state, results, epoch)
            test_callback(
                config, training_state, test_state,
                model=model, optimizer=optimizer, scheduler=scheduler,
                summary_writer=summary_writer
            )
    
    return model


def parse_args(args=None):
    """Parse command line arguments"""
    parser = ArgumentParser('train_polar_bimamba')
    parser.add_argument('--code-hint', dest='code_hint', type=str, required=True,
                       help="Code hint e.g., POLAR_N32_K16")
    parser.add_argument('--path', dest='path', default='results_bimamba', required=False,
                       help="Path for results [Default: results_bimamba]")
    parser.add_argument('--config-file', dest='config', type=str, default='train_config.json',
                       required=False, help="Path to config file")
    parser.add_argument('--epochs-per-test', dest='epochs_per_test', type=int, default=10,
                       required=False, help="Epochs between tests [Default: 10]")
    parser.add_argument('--simple', action='store_true',
                       help="Use SimplePolarBiMambaDecoder instead of masked version")
    return parser.parse_args(args=args)


def load_train_config(config_path):
    """Load training config from JSON"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


DEFAULT_PARAMETERS = dict(
    code_hint="POLAR_N32_K16",
    d_model=128,
    d_state=128,
    N_dec=8,
    warmup_lr=1.0e-3,
    warmup_length=10,
    epochs=20000,
    eta_min=1e-10,
    batch_size=128,
    seq_len=32,
)


def get_next_dir(path):
    """Get next run directory"""
    try:
        _, runs, _ = next(os.walk(path))
        runs = tuple(filter(lambda name: name.startswith('run'), runs))
        i = len(runs)
    except StopIteration:
        i = 0
    return os.path.join(path, f'run_{i}')


def main():
    """Main training function"""
    args = parse_args()
    training_config = load_train_config(args.config)
    
    parameters = {
        **DEFAULT_PARAMETERS,
        **training_config,
        'code_hint': args.code_hint
    }
    
    path = get_next_dir(args.path)
    
    # Choose model class
    model_cls = SimplePolarBiMambaDecoder if args.simple else PolarBiMambaDecoder
    
    config, model, optimizer, training_state, dataset, summary_writer = \
        initialize(path, model_cls=model_cls, **parameters)
    
    model = train_model(
        config,
        model,
        optimizer,
        training_state,
        dataset,
        summary_writer,
        epochs_per_test=args.epochs_per_test
    )
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
