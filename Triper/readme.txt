For training run


python train_multi_label.py --data [location of data] --log [name of log folder] > train_log.txt

Extract the best validation epoch from the log train_log.txt


for test
python test.py --data [location of data] --weight [location to the weight] --type multi