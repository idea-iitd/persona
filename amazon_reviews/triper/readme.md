## Training 
``` bash
python train_multi_label.py --data [location of data] --log [name of log folder] > train_log.txt
  ```
**example**
```bash 
python train_multi_label.py --data data/ --log log_v1 > train_log.txt
```

**Extract the best validation epoch from the log train_log.txt**



## Testing
```bash
python test.py --data [location of data] --weight [location to the weight] --type multi
```
**example**
```bash
python test.py --data data/ --weight logs/log/params_at_epochXXX.pth  --type multi
bash
