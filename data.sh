for i in train-clean-100/ test-clean/ dev-clean/
do
  for j in $(ls $i) # 1271
  do
    for k in $(ls $i/$j) # 128104
    do
      sed -i 's/<VP> //g' $i/$j/$k/*.normalized.txt
      sed -i 's/ <\/VP>//g' $i/$j/$k/*.normalized.txt
    done
  done
done
