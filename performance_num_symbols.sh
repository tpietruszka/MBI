#!/usr/bin/env bash

model_folder="./test_files/"
length=10000

echo "num_symbols r_exec_time python_exec_time"
for num_symbols in 4 8 12 16
do
    model_path=$model_folder$num_symbols"_states_model.csv"
    sequence_file=$model_folder"/"$num_symbols"_symbols_sequence_"$length".csv"
    Rscript generate_sequence.R $model_path $length --quiet > $sequence_file 2>/dev/null
    r_time=$( (\time -f "%U" Rscript viterbi_decode.R $model_path --quiet < $sequence_file > /dev/null ) 2>&1)
    python_time=$( (\time -f "%U" ./main.py $model_path --quiet < $sequence_file > /dev/null) 2>&1)
    echo "$num_symbols $r_time $python_time"
done

