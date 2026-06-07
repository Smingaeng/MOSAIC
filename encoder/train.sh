for i in {1..5}
do
   echo "Running iteration $i..."
   python train_encoder.py --model lightgcn_gene --dataset yelp --cuda 0
done

echo "All iterations completed."