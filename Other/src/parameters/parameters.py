from dataclasses import dataclass

@dataclass
class Parameters:
   # Preprocessing parameeters
   seq_len: int = 250
   num_words: int = 50000
   
   # Model parameters
   embedding_size: int = 100
   out_size: int = 248
   stride: int = 1
   
   # Training parameters
   epochs: int = 10
   batch_size: int = 64
   learning_rate: float = 0.001

   target = .85
   num_exploitation = 32
   total_num = 2*num_exploitation
   alpha = 1
   active_learning = True

   cosine = True