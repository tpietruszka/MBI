#!/usr/bin/env bash

model_folder="./test_files/"
model_name="coin_toss"

model_path="$model_folder$model_name""_model.csv"

echo "sequence_length r_exec_time python_exec_time"
for length in {10000..100000..10000}
do
    sequence_file=$model_folder"/"$model_name"_sequence_"$length".csv"
    Rscript generate_sequence.R $model_path $length --quiet > $sequence_file 2>/dev/null
    r_time=$( (\time -f "%U" Rscript viterbi_decode.R $model_path --quiet < $sequence_file > /dev/null ) 2>&1)
    python_time=$( (\time -f "%U" ./main.py $model_path --quiet < $sequence_file > /dev/null) 2>&1)
    echo "$length $r_time $python_time"
done

