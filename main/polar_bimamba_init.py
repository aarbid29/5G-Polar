import json
import os
import torch
from torch.utils.data import DataLoader
import logging
from hashlib import md5
from dataclasses import fields, MISSING
from polar_bimamba_dataset import PolarCodeDataset, EbN0_to_std
from dataclasses import dataclass
from typing import Any
CODES_PATH = os.path.join(os.path.dirname(__file__), "codes")
from polar_bimamba_model import BiMambaConfig, Code

def Read_pc_matrix_alist(fileName):
    """Read parity check matrix from .alist file"""
    import numpy as np
    with open(fileName, 'r') as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(
            lines[0].rstrip('\n'), dtype=int, sep=' ')
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(
                lines[column].rstrip('\n'), dtype=int, sep=' ')
            for row in nonZeroEntries:
                if row > 0:
                    H[row - 1, column - 4] = 1
        return H


def get_generator(pc_matrix_):
    """Compute generator matrix from parity check matrix"""
    import numpy as np
    
    def row_reduce(mat, ncols=None):
        assert mat.ndim == 2
        ncols = mat.shape[1] if ncols is None else ncols
        mat_row_reduced = mat.copy()
        p = 0
        for j in range(ncols):
            idxs = p + np.nonzero(mat_row_reduced[p:,j])[0]
            if idxs.size == 0:
                continue
            mat_row_reduced[[p,idxs[0]],:] = mat_row_reduced[[idxs[0],p],:]
            idxs = np.nonzero(mat_row_reduced[:,j])[0].tolist()
            idxs.remove(p)
            mat_row_reduced[idxs,:] = mat_row_reduced[idxs,:] ^ mat_row_reduced[p,:]
            p += 1
            if p == mat_row_reduced.shape[0]:
                break
        return mat_row_reduced, p
    
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:,pc_matrix.shape[1]:])[0]


def get_generator_and_parity(code, standard_form=False):
    """Load generator and parity check matrices"""
    import numpy as np
    
    n, k = code.n, code.k
    path_pc_mat = os.path.join(CODES_PATH, f'{code.code_type}_N{str(n)}_K{str(k)}')
    
    if code.code_type in ['POLAR', 'BCH']:
        ParityMatrix = np.loadtxt(path_pc_mat + '.txt')
    elif code.code_type in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = Read_pc_matrix_alist(path_pc_mat + '.alist')
    else:
        raise Exception(f'Wrong code {code.code_type}')
    
    GeneratorMatrix = get_generator(ParityMatrix)
    
    assert np.all(np.mod((np.matmul(GeneratorMatrix, ParityMatrix.transpose())), 2) == 0)
    assert np.sum(GeneratorMatrix) > 0
    
    return GeneratorMatrix.astype(float), ParityMatrix.astype(float)


def code_from_hint(hint: str):
    """Create Code object from hint string like 'POLAR_N32_K16'"""
    hint = hint.upper()
    parts = hint.split('_')
    code_type = parts[0]
    code_n = int(parts[1][1:])
    code_k = int(parts[2][1:])
    
    code = Code(code_n, code_k, code_type)
    
    # Load matrices
    G, H = get_generator_and_parity(code, standard_form=True)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    
    return code


def non_default_fields(instance):
    """Get non-default field values from dataclass"""
    diffs = {}
    for field in fields(instance):
        current = getattr(instance, field.name)
        
        if field.default is MISSING and field.default_factory is MISSING:
            diffs[field.name] = current
        elif field.default_factory is not MISSING:
            diffs[field.name] = current
        elif field.default is not MISSING and current != field.default:
            diffs[field.name] = current
    
    return diffs


def code_to_hint(code: Code) -> str:
    """Convert Code to hint string"""
    return f"{code.code_type.upper()}_N{code.n}_K{code.k}"


def dump_config(config: BiMambaConfig):
    """Dump config to dictionary"""
    config_dump = non_default_fields(config)
    config_dump['code_hint'] = code_to_hint(config_dump.pop('code'))
    return config_dump


def config_hash(config):
    """Generate hash for config"""
    config_dict = dump_config(config)
    temp_string = (''.join([f'{k}{v}' for k, v in config_dict.items()])).encode()
    hash_string = md5(temp_string).hexdigest()
    return hash_string.upper()


def create_config(
    output_path=".output",
    code_hint="POLAR_N32_K16",
    **kwargs
):
    """Create BiMamba config"""
    code = code_from_hint(code_hint)
    
    config = BiMambaConfig(
        code=code,
        **kwargs
    )
    
    if config.experiment_type:
        path = os.path.join(output_path, config.experiment_type, config_hash(config))
    else:
        path = os.path.join(output_path, config_hash(config))
    
    config.path = path
    return config


def create_dataset(config: BiMambaConfig):
    """Create train and test dataloaders"""
    code = config.code
    
    # SNR ranges
    EbNo_range_test = range(3, 7)
    EbNo_range_train = range(2, 8)
    
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    
    # Training dataset
    train_dataset = PolarCodeDataset(
        code=code,
        sigma_list=std_train,
        dataset_size=config.batch_size * config.train_batch_count,
        zero_cw=config.zero_cw
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers
    )
    
    # Test datasets (one per SNR)
    test_dataloader_list = []
    for ii in range(len(std_test)):
        test_dataset = PolarCodeDataset(
            code=code,
            sigma_list=[std_test[ii]],
            dataset_size=config.test_batch_size * config.test_batch_count,
            zero_cw=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.workers
        )
        test_dataloader_list.append(test_loader)
    
    return train_dataloader, test_dataloader_list, EbNo_range_train, EbNo_range_test


def save_checkpoint(checkpoint):
    config = checkpoint['config']
    config_file = os.path.join(config.path, 'config.json')
    
    if not os.path.isfile(config_file):
        with open(config_file, 'w') as f:
            json.dump(dump_config(config), f, indent=2)
    
    if 'model' in checkpoint:
        torch.save(checkpoint['model'], os.path.join(config.path, 'model'))
    
    if 'best_model' in checkpoint:
        torch.save(checkpoint['best_model'], os.path.join(config.path, 'best_model'))
    
    if 'optimizer' in checkpoint:
        torch.save(checkpoint['optimizer'], os.path.join(config.path, 'optimizer'))
    
    if 'state' in checkpoint:
        with open(os.path.join(config.path, 'state.json'), 'w') as f:
            json.dump(checkpoint['state'], f, indent=2)


def load_checkpoint(path):
    checkpoint = {}
    
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config_dict.pop('path', None)
    checkpoint['config'] = BiMambaConfig(
        code=code_from_hint(config_dict.pop('code_hint')),
        path=path,
        **config_dict
    )
    
    if os.path.isfile(model_path := os.path.join(path, 'model')):
        checkpoint['model'] = torch.load(model_path)
    
    if os.path.isfile(model_path := os.path.join(path, 'best_model')):
        checkpoint['best_model'] = torch.load(model_path)
    
    if os.path.isfile(optimizer_path := os.path.join(path, 'optimizer')):
        checkpoint['optimizer'] = torch.load(optimizer_path)
    
    if os.path.isfile(state_path := os.path.join(path, 'state.json')):
        with open(state_path) as f:
            checkpoint['state'] = json.load(f)
    
    return checkpoint


def initialize(
    path,
    model_cls,
    optimizer_init=None,
    experiment=None,
    summary=True,
    best=False,
    resume=False,
    **parameters
):
    """Initialize model, optimizer, and training state"""
    from torch.utils.tensorboard import SummaryWriter
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = create_config(output_path=path, **parameters)
    
    if experiment:
        load_path = path
    elif resume:
        load_path = config.path
    else:
        load_path = None
    
    if load_path is None:
        os.makedirs(config.path, exist_ok=True)
        with open(os.path.join(config.path, 'config.json'), 'w') as f:
            json.dump(dump_config(config), f, indent=2)
        checkpoint = {'config': config, 'state': {}}
    else:
        checkpoint = load_checkpoint(load_path)
        config = checkpoint['config']
    
    # Setup logging
    handlers = [
        logging.FileHandler(os.path.join(config.path, 'logging.txt')),
        logging.StreamHandler()
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=handlers
    )
    
    # Create model
    model = model_cls(config=config).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'best_model' in checkpoint and best:
        model.load_state_dict(checkpoint['best_model'])
    
    # Create optimizer
    if optimizer_init is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.warmup_lr)
    else:
        optimizer = optimizer_init(model=model, config=config)
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Training state
    state = checkpoint.get('state', {})
    training_state = {
        'epoch': state.get('epoch', 0),
        'best_loss': state.get('best_loss', float('inf')),
        'best_ber': state.get('best_ber', float('inf')),
        **state
    }
    
    # Summary writer
    summary_writer = None
    if summary:
        summary_writer = SummaryWriter(config.path)
    
    # Dataset
    dataset = create_dataset(config)
    
    logging.info(
        f'Model initialized. '
        f'Parameters: {sum(p.numel() for p in model.parameters())}. '
        f'N_dec={config.N_dec}, d_model={config.d_model}, code={config.code}'
    )
    
    return config, model, optimizer, training_state, dataset, summary_writer
