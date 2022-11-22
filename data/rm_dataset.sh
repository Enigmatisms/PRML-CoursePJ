
for ((i=500;i<2000;i+=500)); do
    folder_name="train_${i}/*";
    folder_name="test_${i}/*";
    rm ${folder_name}
done

echo "All non-label data removed."