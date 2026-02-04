import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import re
import json


def sign_to_bin(x):
    #for BPSK mapping
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    #noise be for a given Eb/N0 and code rate in AWGN channel
    snr = EbN0 + 10.0 * np.log10(2 * rate)
    return np.sqrt(1.0 / (10.0 ** (snr / 10.0)))


def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()


def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()


class PolarCodeDataset(Dataset):
    def __init__(
        self,
        code,
        sigma_list,
        dataset_size,
        zero_cw=True,
        seed=None
    ):
        self.code = code
        self.sigma_list = sigma_list
        self.dataset_size = dataset_size
        self.zero_cw = zero_cw
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)
        
        if zero_cw:
            self.zero_word = torch.zeros((self.code.k)).long()
            self.zero_codeword = torch.zeros((self.code.n)).long()
        else:
            self.zero_word = None
            self.zero_codeword = None
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if self.zero_codeword is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_codeword
        
        # add AWGN noise
        sigma_idx = idx % len(self.sigma_list)
        z = torch.randn(self.code.n) * self.sigma_list[sigma_idx]
        y = bin_to_sign(x) + z
        
        # compute magnitude and syndrome
        magnitude = torch.abs(y)
        syndrome = torch.matmul(
            sign_to_bin(torch.sign(y)).long(),
            self.pc_matrix
        ) % 2
        syndrome = bin_to_sign(syndrome)
        
        return (
            m.float(),
            x.float(),
            z.float(),
            y.float(),
            magnitude.float(),
            syndrome.float()
        )

def get_reliability_seq(N: int, master_reliability_sequence: list):
    rel_seq = []
    count = 0
    
    while len(rel_seq) != N:
        if master_reliability_sequence[count] < N:
            rel_seq.append(master_reliability_sequence[count])
        count += 1
    
    assert len(rel_seq) == N
    return rel_seq


def find_N(message_bits_length, max_n=1024):
    assert message_bits_length != 0 and message_bits_length <= max_n
    
    for i in [32, 64, 128, 256, 512, 1024]:
        if message_bits_length <= i:
            return i
    
    print(f"Error! Message bits length is out of bound: {message_bits_length}")
    return None


def create_channel_input_vector(message_bits, reliability_json_path="reliability_sequences.json"):
    """
    Creates channel input vector and frozen bit prior vector.
    Returns: channel_input_vector, frozen_bits_prior_vector, N
    """
    N = find_N(len(message_bits))
    
    channel_input_vector = [0] * N
    frozen_bits_prior_vector = [1] * N
    
    assert re.fullmatch('[01]+', ''.join(str(i) for i in message_bits))
    
    # load reliability sequence
    with open(reliability_json_path, 'r+') as f:
        data = json.load(f)
        
        if str(N) in data and data[str(N)]:
            reliability_seq = data[str(N)]
        else:
            reliability_seq = get_reliability_seq(N, data["master_list"])
            data[str(N)] = reliability_seq
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        
        frozen_sets = reliability_seq[len(message_bits):]
    
    # place message bits in high reliability positions
    for i in range(len(message_bits)):
        channel_input_vector[reliability_seq[i]] = message_bits[i]
    
    # mark frozen positions
    for i in frozen_sets:
        frozen_bits_prior_vector[i] = 0
    
    return channel_input_vector, frozen_bits_prior_vector, N


def polar_encode(N: int, channel_input_vector: list):
    """Polar encoding using butterfly structure"""
    assert N == len(channel_input_vector)
    n = int(np.log2(N))
    
    x = channel_input_vector.copy()
    for i in range(n):
        step = 2**i
        for j in range(0, N, 2*step):
            for k in range(step):
                x[j+k] ^= int(x[j+k+step])
    
    return np.array(x).astype(int)


def modulation_bpsk(polar_coded_msg: np.ndarray):
    return 1 - 2 * polar_coded_msg


def awgn_channel(modulated_sequence, SNRs_db, message_bit_size, block_length):
    SNRs_db = np.array(SNRs_db)
    code_rate = float(message_bit_size) / float(block_length)
    SNRs_linear = 10**(SNRs_db / 10)
    variances = np.sqrt(1 / (2 * code_rate * SNRs_linear))
    
    noises = []
    for variance in variances:
        noise = np.random.normal(0, variance, size=(block_length))
        noises.append(noise)
    
    noises_np = np.array(noises)
    result = noises_np + modulated_sequence
    result = result.squeeze(0)
    
    return result


def generate_polar_sample(message_bit_size, SNRs_db, reliability_json_path="reliability_sequences.json"):
    msg_sequence = np.random.randint(0, 2, size=message_bit_size)
    
    civ, frozen_bit_prior, N = create_channel_input_vector(
        message_bits=msg_sequence,
        reliability_json_path=reliability_json_path
    )
    
    polar_coded_form = polar_encode(N, civ)
    modulated_signal = modulation_bpsk(polar_coded_msg=polar_coded_form)
    
    channel_observation_vector = awgn_channel(
        modulated_sequence=modulated_signal,
        SNRs_db=SNRs_db,
        message_bit_size=message_bit_size,
        block_length=N
    )
    
    target = civ
    
    return channel_observation_vector, frozen_bit_prior, target

