# reads a sequence of observations from standard input, computes the most probable sequence 
# of underlying states using Viterbi's algorithm

source("load_model.R")


args <- commandArgs(trailingOnly = TRUE)
model_file <- args[1]
if(is.na(model_file)) { # if no model file given, defaulting to the example
  model_file <- "./bigger_model.csv"
}
model = load_model(model_file)

con=file("stdin", "r")
writeLines("Enter the sequence to be decoded (as one line): ")
observations <- scan(file=con, nlines=1, what='character', blank.lines.skip=TRUE)
output <-viterbi(model, observations)
writeLines("Result: ")
cat(output)
writeLines("")
