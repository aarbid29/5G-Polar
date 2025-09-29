import re, json
import numpy as np

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
    """

    str_message_bits = str(message_bits)

    N= find_N(len(str_message_bits))

    channel_input_vector =[0] * N
    frozen_bits_prior_vector =[0] * N
    
   
    if not re.fullmatch('[01]+', str_message_bits):
       print("Error! Message bits need to be binary")
       return
    
    
    with open("reliability_sequences.json", 'r+') as f:
       data = json.load(f)
       
       reliability_seq = data[str(N)]
       frozen_sets =  reliability_seq[len(str_message_bits):]

       if len(reliability_seq)==0: # reliability sequence for that N is not yet computed
          
         reliability_seq = get_reliability_seq(N, data["master_list"])
         data[str(N)] = reliability_seq

         f.seek(0)
         json.dump(data, f, indent=4)
         f.truncate()
         f.close()
    
    for i in range(len(str_message_bits)):
       channel_input_vector[reliability_seq[i]] = int(str(message_bits)[i])
    
    for i in frozen_sets:
       frozen_bits_prior_vector[i]=-1
       
       
    return channel_input_vector, frozen_bits_prior_vector


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

   return x


      

if __name__=="__main__":

  civ, frozen_sets = create_channel_input_vector(message_bits=110101010101110111)
  print(frozen_sets)
  print(f"\n {civ}")

  print(polar_encode(32, civ))
  

