# MBI
## Hidden Markov Chain decoding using Viterbi algorithm - MBI class project

### Includes:
- ViterbiDecoder.py - an implementation of Viterbi algorithm and utility functions for loading and saving underlying Markov's model
- main.py - a simple console-based user interface for ViterbiDecoder
- load_model.R - an R script loading Markov models, used by:
  - viterbi_decode.R - an interface to Viterbi algorithm in the "HMM" R package - used to verify correctness of our implementation
  - generate_sequence.R - an interface to Markov model simulation - used to generate test sequences that can be decoded afterwards


### Instructions:
#### main.py - Usage: 
- ./main.py [model_file_name]

Then a sequence of observations should be provided in one line (to standard input). Observations are sequentially numbered, starting at 0
If no model file was given, user will be prompted to enter all the model parameters manually (and choose a filename to save it).

#### viterbi_decode.R - Usage: 
- Rscript viterbi_decode.R [model_file_name]

Observations loaded the same way

#### generate_sequence.R - Usage: 
- Rscript generate_sequence.R [model_file_name] [sequence_length] 

Outputs a sequence of observations and a sequence of underlying hidden states

### Model file format: 
Space-delimited numerical values, over multiple lines
- First line - 1 value - S - number of all possible states
- Second line - 1 value - O - number of all possible observations
- Third line - S values - probabilities of initial states
- S lines, S values in each - transition probabilities from (row_number) to (col_number) 
- S lines, O values in each - probability of observation (col_number), given state (row_number) 
