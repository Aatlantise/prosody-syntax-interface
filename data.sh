for i in train-clean-100/ test-clean/ dev-clean/
do
  for j in $(ls $i) # 1271
  do
    for k in $(ls $i/$j) # 128104
    do
#      sed -i 's/<VP> //g' $i/$j/$k/*.normalized.txt
#      sed -i 's/ <\/VP>//g' $i/$j/$k/*.normalized.txt
      for file in $(ls $i/$j/$k/*.original.txt)
      do
        mv $file $file.temp
      done
    done
  done
done
