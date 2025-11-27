import re, json
import numpy as np
import pandas as pd

N_MAX = 1024
K_START = 12



def get_reliability_seq(N: int, master_reliability_sequence: list):
   
   
   """Returns a reliability sequence for a given blocklength N"""

   rel_seq = []
   count=0

   while len(rel_seq)!=N:
   #   print("count is ", count)
      rel_seq.append(master_reliability_sequence[count]) if master_reliability_sequence[count]<N else None
      count+=1

   assert(len(rel_seq)==N)
   
   return rel_seq


def find_N(message_bits_length):

   """
   Given length of message bits to be encoded, finds the appropriate block length N 
   """

   assert(message_bits_length!=0 & message_bits_length<=N_MAX)

   for i in [32, 64, 128, 512, 1024]:
      if message_bits_length <= i:
         return i
      
   print(f"Error! Message bits length is out of bound: {message_bits_length}")
   return
      



def create_channel_input_vector(message_bits):

    """
    Takes in message bits and returns the channel input vector (u), frozen bit prior vector (Akc)

    u is the vector consisting of message bits in positions with high reliability. Rest are frozen bits.
    """

    N= find_N(len(message_bits))

    channel_input_vector =[0] * N
    frozen_bits_prior_vector =[0] * N
    
   
    assert(re.fullmatch('[01]+', ''.join(str(i) for i in message_bits)))
    
    
    with open("reliability_sequences.json", 'r+') as f:
       data = json.load(f)
       
       reliability_seq = data[str(N)]
       frozen_sets =  reliability_seq[len(message_bits):]

       if len(reliability_seq)==0: # reliability sequence for that N is not yet computed
          
         reliability_seq = get_reliability_seq(N, data["master_list"])
         data[str(N)] = reliability_seq

         f.seek(0)
         json.dump(data, f, indent=4)
         f.truncate()
         f.close()
    
    for i in range(len(message_bits)):
       channel_input_vector[reliability_seq[i]] = message_bits[i]
    
    for i in frozen_sets:
       frozen_bits_prior_vector[i]=1
       
       
    return channel_input_vector, frozen_bits_prior_vector, N


def polar_encode(N: int, channel_input_vector:list):

   """
   Returns the polar encoded version of channel_input_vector using butterfly loop
   """

   assert(N==len(channel_input_vector))
   n = int(np.log2(N))

   x = channel_input_vector.copy()
   for i in range(n):
        step = 2**i
        for j in range(0, N, 2*step):
            for k in range(step):
                x[j+k] ^= int(x[j+k+step])  # XOR operation

   return np.array(x).astype(int)


def modulation_bpsk(polar_coded_msg: np.ndarray):
    """
    polar_coded_msg: numpy array of 0/1 bits
    returns: numpy array of BPSK symbols (+1/-1)
    """
    return 1 - 2 * polar_coded_msg



def one_hot(msg_bit_sequence: int):

   """
   Encodes the message bit sequence into a one hot encoded format.
   """

   
   assert re.fullmatch('[01]+', ''.join(str(i) for i in msg_bit_sequence)), "Sequence must contain only 0 or 1"

   seq_array = np.array([b for b in msg_bit_sequence], dtype=int)
   one_hot = np.zeros((len(seq_array), 2), dtype=int)
   one_hot[np.arange(len(seq_array)), seq_array] = 1 # fancy way of creating one hot encoding: [0, 1, 2], [0, 1, 0] => insert 1 at one_hot[0,0], one_hot[1,1] and so on
   return one_hot
   


def one_hot_smoothing(msg_bit_sequence: int, num_classes=2, smoothing_factor=0.1):

   """
   Converts the given binary sequence into a list of one hot smoothed vectors.
   """

   one_hot_encoded = one_hot(msg_bit_sequence)
   smoothed = (1 - smoothing_factor) * one_hot_encoded + (smoothing_factor / 2)
   return np.round(smoothed, 3)


def awgn_channel(modulated_sequence:list, SNRs_db:list, message_bit_size:int, block_length:int):

   """
   Takes in a modulated sequence and list of SNR values in db (both 1D).

   Returns a list of lists of the AWGN channel outputs on the various SNR values defined in SNRs_db list
   """

   SNRs_db = np.array(SNRs_db)

   code_rate = float(message_bit_size)/float(block_length)

   SNRs_linear = 10**(SNRs_db/10)

   variances = np.sqrt(1/(2*code_rate*SNRs_linear))


   noises = []

   for variance in variances:
      noise = np.random.normal(0, variance, size=( block_length))
     # print(noise)
      noises.append(noise)
   
   noises_np = np.array(noises)
 #  print(noises_np.shape)

   result = noises_np + modulated_sequence
   result = result.squeeze(0)

   return result




def generate_data(message_bit_size, SNRs_db):

   """
   Generates a single data instance for given message bit size.

   """

   msg_sequence = np.random.randint(0, 2, size=message_bit_size)
 #  target = one_hot_smoothing(msg_sequence, smoothing_factor=smoothing_factor)
   
   

   civ, frozen_bit_prior, N = create_channel_input_vector(message_bits=msg_sequence)
   polar_coded_form = polar_encode(N, civ)
   modulated_signal = modulation_bpsk(polar_coded_msg=polar_coded_form)

   channel_observation_vector = awgn_channel(modulated_sequence=modulated_signal, SNRs_db=SNRs_db, message_bit_size=message_bit_size, block_length=N)
   target = civ

  # print(f"Shape of channel observation vector: {channel_observation_vector.shape}")
   

   return channel_observation_vector, frozen_bit_prior, target



def generate_dataset(message_bit_size, SNRs_db, smoothing_factor, num_samples):

   """
   Generates a complete dataset(dataframe) of size num_samples for a Polar Code Decoder

   Features:  Channel Observation Vector (varying SNRs as specified by SNRs_db), Frozen bit prior vector, 
   Target: Message bits (1 hot smoothed)
   Extras: Original message sequence
   """

   columns = [f'channel_ob_vector_snr_{i}' for i in SNRs_db] + ['frozen_bit_prior', 'original_msg', 'target']

   dataset = []

   for i in range(num_samples):

      print(f"Generating {i}th sample...\n")

      channel_observation_vector, frozen_bit_prior, target, msg_sequence = generate_data(message_bit_size, SNRs_db, smoothing_factor)
      instance = [channel_observation_vector[i] for i in range(len(SNRs_db))] + [frozen_bit_prior, msg_sequence, target]
      dataset.append(instance)
   
   print("Generation completed!")
   
   return pd.DataFrame(dataset, columns=columns)





if __name__=="__main__":

    dataframe = generate_dataset(message_bit_size=8, SNRs_db=[4, 4.5, 5, 5.5, 6], smoothing_factor=0.1, num_samples=256000)
    dataframe.to_csv("data_32bits_polar.csv")
    
    dataframe_2 = generate_dataset(message_bit_size=16, SNRs_db=[4, 4.5, 5, 5.5, 6], smoothing_factor=0.1, num_samples=372000)
    dataframe_2.to_csv('data_32bits_polar.csv', mode='a', index=False, header=False)

    dataframe_3 = generate_dataset(message_bit_size=24, SNRs_db=[4, 4.5, 5, 5.5, 6], smoothing_factor=0.1, num_samples=372000)
    dataframe_3.to_csv('data_32bits_polar.csv', mode='a', index=False, header=False)


  


  

  

  

