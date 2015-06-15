# generates a sequence of observations, and a sequence of underlying states from a given hidden markov model.
# command line arguments: model file name, length of the sequence to be generated, optional "quiet" flag

source("load_model.R")

args <- commandArgs(trailingOnly = TRUE)
model_file <- args[1]
sequence_length <- args[2]
quiet <- args[3] # no user prompts - for batch processing

if(is.na(model_file) || is.na(sequence_length)) {
  stop("Usage: Rscript generate_sequence.R [model_file] [sequence_length] [--quiet")
}
sequence_length <- strtoi(sequence_length, 10)

only_observations = FALSE
if(!is.na(quiet) && quiet == "--quiet") {
  only_observations = TRUE
}
  

model = load_model(model_file)
result = simHMM(model, sequence_length)
if(!only_observations) { 
  writeLines("States: ")
  cat(result$states)
  writeLines("\nObservations: ")
}
cat(result$observation)
writeLines("")