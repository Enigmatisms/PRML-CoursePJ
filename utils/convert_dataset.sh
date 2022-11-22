
current_folder=`pwd | xargs echo`
echo "Start to convert dataset... Current folder: ${current_folder}"

for ((i=500;i<2000;i+=500)); do
    echo "Converting: ${i}"
    python3 ./utils.py --atcg_seq_len ${i};
    python3 ./utils.py --atcg_seq_len ${i} --is_test;
    python3 ./utils.py --cvt_label --label_seq_id ${i};
    python3 ./utils.py --cvt_label --label_seq_id ${i} --is_test;
    echo "Sequence length: ${i} conversion completed"
done