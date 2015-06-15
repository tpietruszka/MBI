# loads a hidden markov model from a file

if(!require('HMM')) {
  install.packages('HMM')
}
library('HMM')

load_model <- function(model_file) {
  model_data <- read.delim(model_file, header=FALSE, sep=' ')
  S <- model_data[1, 1]
  O <- model_data[2, 1]
  p_init <- unlist(model_data[3,])
  p_transition <-matrix(unlist(model_data[4:(4+S-1), 1:(S)]), nrow=S)
  p_observation <- matrix(unlist(model_data[(4+S):(4+S+S-1), (1:O)]), nrow=S)
  
  state_names <- sapply(seq(0, S-1), toString) # using '0', '1', ..., 'S-1' as state names. Observations named the same way
  observation_names <- sapply(seq(0, O-1), toString)
  
  model <- initHMM(state_names, observation_names, startProbs=p_init, transProbs=p_transition, emissionProbs=p_observation)
  return(model)
}