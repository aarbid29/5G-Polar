import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import json
import os
from argparse import ArgumentParser

from polar_bimamba_dataset import BER, FER, PolarCodeDataset, EbN0_to_std
from polar_bimamba_init import initialize, load_checkpoint
from polar_bimamba_model import PolarBiMambaDecoder, SimplePolarBiMambaDecoder


TEST_BATCH_SIZE = 512


def test_model(model, device, test_loader_list, EbNo_range_test):
    """Test model and compute BER/FER"""
    model.eval()
    results = {}
    total_ber = 0
    code_length = 1
    
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_fer = cum_count = 0.0
            
            with tqdm(
                total=len(test_loader.dataset),
                unit='codewords',
                unit_scale=True,
                desc=f"Testing {EbNo_range_test[ii]} dB"
            ) as pbar:
                for m, x, z, y, magnitude, syndrome in test_loader:
                    code_length = x.shape[1]
                    z_pred = model(magnitude.to(device), syndrome.to(device))
                    x_pred = model.get_codeword(z_pred, y.to(device))
                    
                    test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                    test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                    cum_count += x.shape[0]
                    pbar.update(x.shape[0])
                    
                    # Stop after 100 errors for efficiency
                    if test_fer >= 100:
                        break
            
            test_ber /= cum_count
            test_fer /= cum_count
            ln_ber = -np.log(test_ber) if test_ber > 0 else float('inf')
            
            logging.info(
                f'Test EbN0={EbNo_range_test[ii]} dB: '
                f'BER={test_ber:.2e}, -ln(BER)={ln_ber:.2e}, '
                f'FER={test_fer:.2e}, TotalBits={cum_count * code_length}'
            )
            
            results[f"BER_{EbNo_range_test[ii]}"] = test_ber
            results[f"FER_{EbNo_range_test[ii]}"] = test_fer
            total_ber += test_ber / len(test_loader_list)
    
    results['test_ber'] = total_ber
    return results


def _test(config, model):
    """Test on standard SNR range"""
    EbNo_range_test = range(3, 7)  # Test on 3, 4, 5, 6 dB
    
    code = config.code
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    
    # Create test datasets
    test_dataloader_list = []
    for ii in range(len(std_test)):
        test_dataset = PolarCodeDataset(
            code=code,
            sigma_list=[std_test[ii]],
            dataset_size=TEST_BATCH_SIZE * 10000,  # Large dataset
            zero_cw=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )
        test_dataloader_list.append(test_loader)
    
    return test_model(model, 'cuda', test_dataloader_list, EbNo_range_test)


def load_path(path, best=False, model_cls=PolarBiMambaDecoder):
    """Load model from path"""
    checkpoint = load_checkpoint(path)
    config = checkpoint['config']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_cls(config=config).to(device)
    
    if best and 'best_model' in checkpoint:
        model.load_state_dict(checkpoint['best_model'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError("No model found in checkpoint")
    
    return config, model


def find_experiments(test_result_dir):
    """Find all experiment directories with config.json"""
    experiments = set()
    for path, dirs, files in os.walk(test_result_dir):
        if 'config.json' in files:
            experiments.add(path)
    return experiments


def validate(path, model_cls=PolarBiMambaDecoder):
    """Validate all experiments in path"""
    experiments = find_experiments(path)
    
    for experiment in sorted(experiments):
        results = {}
        
        # Test regular and best model
        options = [
            {'best': False},
            {'best': True},
        ]
        
        for kwargs in options:
            key = f"{experiment},{kwargs}"
            
            if key in results:
                continue
            
            print(f"\nValidating: {experiment}")
            print(f"Options: {kwargs}")
            
            try:
                config, model = load_path(experiment, model_cls=model_cls, **kwargs)
                results[key] = _test(config, model)
            except Exception as err:
                print(f"Failed: {experiment}, error: {err}")
                continue
            
            # Save results
            with open(os.path.join(experiment, 'validation.json'), 'w') as f:
                json.dump(results, f, indent=2)
    
    print("\nValidation completed!")


def parse_args():
    """Parse arguments"""
    parser = ArgumentParser('validate_polar_bimamba')
    parser.add_argument('--path', dest='path', type=str, required=True,
                       help="Path to experiment directory")
    parser.add_argument('--simple', action='store_true',
                       help="Use SimplePolarBiMambaDecoder")
    return parser.parse_args()


def main():
    """Main validation function"""
    args = parse_args()
    
    model_cls = SimplePolarBiMambaDecoder if args.simple else PolarBiMambaDecoder
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    print(f"Starting validation for: {args.path}")
    validate(args.path, model_cls=model_cls)


if __name__ == "__main__":
    main()
