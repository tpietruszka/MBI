#!/usr/bin/env python3

import numpy as np

class ViterbiDecoder(object):
    """ 
    Decoding hidden markov chains - finding the most likely sequence of hidden states, 
    given a sequence of observed events - using Viterbi algorithm
    The decoder needs all parameters of the model: probabilities of initial states, probabilities 
    of transition between states, and probabilities of each observation for each state. 
    """
    def __init__(self, p_initial, p_transition, p_observation):
        """
        p_initial - vector of probabilities of initial states. Shape: (num_states,)
        p_transition - transition probabilities matrix, rows correspond to initial states, columns - final ones
                        Shape: (num_states, num_states)
        p_observation - emission probability for each observation, given each state. 
                        Rows correspond to states, colums - to observations. Shape: (num_states, num_observations)  
        
        """
        self.p_initial=p_initial
        self.p_transition = p_transition
        self.p_observation = p_observation
        self.num_states = p_initial.shape[0]
        self.num_observations = p_observation.shape[0]
        
        assert self.p_transition.shape[0] == self.p_transition.shape[1] == self.num_states
        assert self.p_observation.shape[0] == self.num_states # TODO: raise exceptions instead
        
    def decode(self, observations):
        viterbi = np.zeros((self.num_states, len(observations))) # probabilities of the most probable sequence leading to a state
        backpointers = np.zeros((self.num_states, len(observations))) # zero-indexed numbers of states chosen in each step
        viterbi[:, 0] = self.p_observation[:, observations[0]] * self.p_initial
        backpointers[:, 0] = np.argmax(viterbi[:, 0])
        
        for t in range(1, len(observations)):
            for s in range(0, self.num_states): # TODO: re-write this loop as matrix operations?
                possibilities = viterbi[:, t-1] * self.p_transition[:, s] * self.p_observation[s, observations[t]]
                viterbi[s, t] = np.max(possibilities)
                backpointers[s, t] = np.argmax(possibilities)
        path = np.zeros(len(observations))
        path[-1] = np.argmax(viterbi[:, -1])
        for t in range(len(observations)-1, 0, -1):
            path[t-1] = backpointers[path[t], t]
        return path
        

        
        

pi = np.array([0.04, 0.02, 0.06, 0.04, 0.11, 0.11, 0.01, 0.09, 0.03, 0.05, 0.06, 0.11, 0.05, 0.11, 0.03, 0.08])
trans = np.array([ \
    [0.08, 0.02, 0.10, 0.05, 0.07, 0.08, 0.07, 0.04, 0.08, 0.10, 0.07, 0.02, 0.01, 0.10, 0.09, 0.01], \
    [0.06, 0.10, 0.11, 0.01, 0.04, 0.11, 0.04, 0.07, 0.08, 0.10, 0.08, 0.02, 0.09, 0.05, 0.02, 0.02], \
    [0.08, 0.07, 0.08, 0.07, 0.01, 0.03, 0.10, 0.02, 0.07, 0.03, 0.06, 0.08, 0.03, 0.10, 0.10, 0.08], \
    [0.08, 0.04, 0.04, 0.05, 0.07, 0.08, 0.01, 0.08, 0.10, 0.07, 0.11, 0.01, 0.05, 0.04, 0.11, 0.06], \
    [0.03, 0.03, 0.08, 0.10, 0.11, 0.04, 0.06, 0.03, 0.03, 0.08, 0.03, 0.07, 0.10, 0.11, 0.07, 0.03], \
    [0.02, 0.05, 0.01, 0.09, 0.05, 0.09, 0.05, 0.12, 0.09, 0.07, 0.01, 0.07, 0.05, 0.05, 0.11, 0.06], \
    [0.11, 0.05, 0.10, 0.07, 0.01, 0.08, 0.05, 0.03, 0.03, 0.10, 0.01, 0.10, 0.08, 0.09, 0.07, 0.02], \
    [0.03, 0.02, 0.16, 0.01, 0.05, 0.01, 0.14, 0.14, 0.02, 0.05, 0.01, 0.09, 0.07, 0.14, 0.03, 0.01], \
    [0.01, 0.09, 0.13, 0.01, 0.02, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.06, 0.11, 0.06, 0.03, 0.14], \
    [0.09, 0.03, 0.04, 0.05, 0.04, 0.03, 0.12, 0.04, 0.07, 0.02, 0.07, 0.10, 0.11, 0.03, 0.06, 0.09], \
    [0.09, 0.04, 0.06, 0.06, 0.05, 0.07, 0.05, 0.01, 0.05, 0.10, 0.04, 0.08, 0.05, 0.08, 0.08, 0.10], \
    [0.07, 0.06, 0.01, 0.07, 0.06, 0.09, 0.01, 0.06, 0.07, 0.07, 0.08, 0.06, 0.01, 0.11, 0.09, 0.05], \
    [0.03, 0.04, 0.06, 0.06, 0.06, 0.05, 0.02, 0.10, 0.11, 0.07, 0.09, 0.05, 0.05, 0.05, 0.11, 0.08], \
    [0.04, 0.03, 0.04, 0.09, 0.10, 0.09, 0.08, 0.06, 0.04, 0.07, 0.09, 0.02, 0.05, 0.08, 0.04, 0.09], \
    [0.05, 0.07, 0.02, 0.08, 0.06, 0.08, 0.05, 0.05, 0.07, 0.06, 0.10, 0.07, 0.03, 0.05, 0.06, 0.10], \
    [0.11, 0.03, 0.02, 0.11, 0.11, 0.01, 0.02, 0.08, 0.05, 0.08, 0.11, 0.03, 0.02, 0.10, 0.01, 0.11]])
obs = np.array([[0.01,0.99], \
                [0.58,0.42], \
                [0.48,0.52], \
                [0.58,0.42], \
                [0.37,0.63], \
                [0.33,0.67], \
                [0.51,0.49], \
                [0.28,0.72], \
                [0.35,0.65], \
                [0.61,0.39], \
                [0.97,0.03], \
                [0.87,0.13], \
                [0.46,0.54], \
                [0.55,0.45], \
                [0.23,0.77], \
                [0.76,0.24]])

# TODO: replace this test data, perhaps move to doctest?

test1 = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
expected1 = np.array([11, 10, 15, 10, 15, 10, 0, 10, 0, 0])
test2 = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
expected2 = np.array([5, 14, 10, 15, 0, 0, 14, 10, 15, 10])
d = ViterbiDecoder(pi, trans, obs)
result1 = d.decode(test1)
result2 = d.decode(test2)
print(result1)
print(result2)
assert(all(result1 == expected1))
assert(all(result2 == expected2))
print("Test succesful")
