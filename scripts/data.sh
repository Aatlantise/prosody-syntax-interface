for i in train-clean-100 test-clean dev-clean
do
  for j in $(ls $i) # 1271
  do
    for k in $(ls $i/$j) # 128104
    do
#      cp ~/data/LibriTTSNPVP/$i/$j/$k/*.syntactic.txt ~/data/LibriTTSNP/$i/$j/$k/
#      sed -i 's/<VP> //g' $i/$j/$k/*.syntactic.txt
#      sed -i 's/ <\/VP>//g' $i/$j/$k/*.syntactic.txt
      for file in $(ls $i/$j/$k/*.syntactic.txt*)
      do
	      new_file="${file%.syntactic.txt}.original.txt"

        # Rename the file
        mv "$file" "$new_file"
      done
    done
  done
done
