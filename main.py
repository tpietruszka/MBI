#!/usr/bin/env python3
from ViterbiDecoder import ViterbiDecoder
import argparse
import sys
parser = argparse.ArgumentParser(description='Decode a hidden markov chain using Viterbi\'s algorithm')
parser.add_argument('model_file_path', nargs='?', type=str, help='A file with model model parameters. If not given, user will be prompted to enter the parameters manually')
args = parser.parse_args()

if(args.model_file_path):
    model = ViterbiDecoder.load_model(args.model_file_path)
else:
    model = ViterbiDecoder()
    print("Enter the path to save the model (leave empty to skip saving): ")
    model_save_path = sys.stdin.readline().strip()
    if(len(model_save_path) > 0):
        model.save_model(model_save_path)

print("Enter the sequence to be decoded (as one line): \n")
observations = ViterbiDecoder.read_number_line(sys.stdin, int, normalize=False)
decoded_states = model.decode(observations)


string_of_int = lambda x: str(int(x))
output = ViterbiDecoder.format_number_line(decoded_states, string_of_int)
print(output)
    
