For training run


python train_multi_label.py --data [location of data] --log [name of log_folder] > train_log.txt
weights are saved inside logs/[log_folder]/
Extract the best validation epoch from the log train_log.txt


for test
python test.py --data [location of data] --weight [location to the weight] --type multi