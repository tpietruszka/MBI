# reads a sequence of observations from standard input, computes the most probable sequence 
# of underlying states using Viterbi's algorithm

source("load_model.R")


args <- commandArgs(trailingOnly = TRUE)
model_file <- args[1]
quiet <- args[2]
if(is.na(model_file)) { # if no model file given, defaulting to the example
  stop("Usage: Rscript viterbi_decode.R [model_file] [--quiet]")
}

only_results = FALSE # skip prompts for batch processing
if(!is.na(quiet) && quiet == "--quiet") {
  only_results = TRUE
}

model = load_model(model_file)

con=file("stdin", "r")
if(! only_results) { writeLines("Enter the sequence to be decoded (as one line): ") }
observations <- scan(file=con, nlines=1, what='character', blank.lines.skip=TRUE, quiet=TRUE)
output <-viterbi(model, observations)
if(! only_results) { writeLines("Result: ") }
cat(output)
writeLines("")
