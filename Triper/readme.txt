
##You can recreate our Conda environment with
conda env create -f environment.yml

#Unzip data.zip folder

unzip data.zip


#For training run


python train_multi_label.py --data [location of data] --log [name of log folder] > train_log.txt

#example
python train_multi_label.py --data data/ --log log/ > train_log.txt

#Extract the best validation epoch from the log train_log.txt



#for test
python test.py --data [location of data] --weight [location to the weight] --type multi

#example 
python test.py --data data/ --weight logs/log/params_at_epochXXX.pth  --type multi