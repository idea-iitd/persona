
"""
File adapted from https://github.com/ds4dm/learn2branch
"""
import warnings
import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
import torch
import random
from datetime import datetime
random.seed(0)
#import pdb
import matplotlib.pyplot as plt

import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborSampler
import tqdm
from torch_geometric.loader import NeighborLoader
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, hamming_loss
from sklearn.metrics import f1_score, confusion_matrix, classification_report, multilabel_confusion_matrix


warnings.filterwarnings("ignore")
labels_columns = [
    'fashion', 'budget', 'sport', 
    'luxury', 'professional', 
    'casual/comfort', 'adventure', 
    'children', 'wedding'
]



def validate(loader, model, comb_embeddingv, true_labels, type = 'single', test=False):
    criterion = nn.CrossEntropyLoss()
    valid_loss = 0

    for batch in enumerate(loader):

        b              = batch[0]
        users          = batch[1][1][:true_labels.shape[0]] 
        product        = batch[1][1][true_labels.shape[0]:]
        edge_ind       = batch[1][2][0]
        edge_emb_index = batch[1][2][1]
        

        
        batch_user_emb = comb_embeddingv[users]
        
        edge_features  = eev[edge_emb_index]
        # pdb.set_trace()

        batch_prod_emb = comb_embeddingv[product]
        
        # ----------------------------------------------------------------------
        batched_states = batch_user_emb.cuda(), edge_ind.cuda(), edge_features.cuda(), batch_prod_emb.cuda()
        #-----------------------------------------------------------------------

            
        
        
        
        predictions = model(batched_states)  # eval mode
        valid_loss = criterion(predictions.cuda(), torch.tensor(true_labels, dtype=torch.float).cuda())
        
        model.eval()
                
        
        
        # #pdb.set_trace()
        if type == 'single':
            predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            true_labels = np.argmax(true_labels, axis=1)
            # pdb.set_trace()
            
            # total = len(true_labels)
            # correct = np.sum(predictions == true_labels)
            # # pdb.set_trace()
            # accuracy = correct / total
            # pdb.set_trace()
            
            f1_micro      = f1_score(true_labels, predictions, average='micro')
            f1_macro      = f1_score(true_labels, predictions, average='macro')
            f1_weighted   = f1_score(true_labels, predictions, average='weighted')
            
            jaccard_macro = jaccard_score(true_labels, predictions, average='macro')
            jaccard_weighted = jaccard_score(true_labels, predictions, average='weighted')
            
            hamming_loss_ = hamming_loss(true_labels, predictions) 
           
            print(f'Test F1 score macro: {f1_macro}')
            print(f'Test F1 score micro: {f1_micro}')
            print(f'Test F1 score weighted: {f1_weighted}')
            print(f'Jaccard macro: {jaccard_macro}')
            print(f'Jaccard weighted: {jaccard_weighted}')
            
            print(f'Hamming: {hamming_loss_}')

        else:

            y_true = []
            y_pred = []
            predictions = predictions.round()
            # pdb.set_trace(k)
            y_true.extend(true_labels)
            y_pred.extend(predictions.detach().cpu().numpy())
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Calculate final metrics
            #pdb.set_trace()
            
            final_f1_macro      = f1_score(y_true, y_pred, average='macro')
            final_f1_weighted   = f1_score(y_true, y_pred, average='weighted')
            final_f1_samples    = f1_score(y_true, y_pred, average='samples')     #known to be good for multilabel

            jaccard_macro    = jaccard_score(y_true, y_pred, average='macro')
            jaccard_weighted = jaccard_score(y_true, y_pred, average='weighted')
            jaccard_samples  = jaccard_score(y_true, y_pred, average='samples')  #known to be good for multilabel

            hamming_loss_ = hamming_loss(y_true, y_pred)


            print(f'Test F1 score macro: {final_f1_macro}')
            print(f'Test F1 score weighted: {final_f1_weighted}')
            print(f'Test F1 score samples: {jaccard_samples}')
            
            print(f'Jaccard macro: {jaccard_macro}')
            print(f'Jaccard weighted: {jaccard_weighted}')
            print(f'Jaccard samples: {jaccard_samples}')
            
            print(f'Hamming: {hamming_loss_}')
            
            # conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
        

            return final_f1_macro, final_f1_weighted
            
    



        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='GAT_baseline_torch',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--data',
        help='name of the folder where train and valid folders are present. Assumes `data/samples` as default.',
        type=str,
        default="data/samples_1.0",
    )
    parser.add_argument(
        '--l2',
        help='value of l2 regularizer',
        type=float,
        default=0.00001
    )
    
    parser.add_argument(
        '--epochs',
        help='',
        type=int,
        default=1000
    )

    
    parser.add_argument(
        '--batch_size',
        help='batch size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--weight',
        help='location of weights',
        type=str,
        default=8
    )

    parser.add_argument(
        '--type',
        help='single/multi',
        type=str,
        required = True,
        default=100
    )
   
    
    import pdb,copy
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    
    

    
    

    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))


    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    from model import *
    
    model = GATPolicy().cuda()
    
    


    
    


    user = 306
    #-------------------------------------------------------------------------------------------------------

    uev       = (torch.empty(user,12))
    # nn.init.xavier_uniform_(uev)
    pev       = torch.tensor(np.load(args.data +'test/pe.npy'), dtype=torch.float32 )
    
    
    eiv       = np.load(args.data +'test/ei_u2pro.npy',allow_pickle=True).T   #2,100
    eev       = (torch.empty(eiv.shape[1],768))
    # nn.init.xavier_uniform_(eev)

    
    


    zeros_array = np.zeros((uev.shape[0], 1867))
    uev = np.hstack((uev, zeros_array))

    if np.min(eiv[1])==0:
        eiv[1] += uev.shape[0]

    
    from torch_geometric.utils import to_undirected
    eiv = torch.tensor(eiv, dtype= torch.long)
    eiv, eev = to_undirected(eiv, eev)
    labelsv = np.load(args.data + 'test/lab.npy')
    # ##pdb.set_trace()


    comb_embeddingv = torch.tensor( np.concatenate( (uev, pev), axis = 0 ) )


    #---------------------------------------------------------------------------------------------------------

 

    datav = Data(x=comb_embeddingv, ei=eiv.long(), edge_attr= (eev),y= (labelsv))  

    l2 = [i for i in range(uev.shape[0])]       
    val_mask = torch.tensor(l2,dtype=torch.long)
    datav.train_idx =  val_mask
    val_loader = NeighborSampler(eiv, [-1], node_idx=val_mask,
                                 batch_size=len(val_mask), shuffle=False,
                                num_workers=10)


    num_neighbors = [-1] 
    # batch_size = args.batch_size        
    # input_nodes = data.train_idx  
    shuffle = False            
    replace = False   
    
    best_f1 = 0
    epoch_for_best_f1 = ''

    model.eval()
    
    # files =  os.listdir(args.weight)

    # for file in files:
    #     weights = torch.load(args.weight + file)
    #     try:
    #         model.load_state_dict(weights)
    #     except:
    #         continue

    #     if args.type == 'single':

    #         f1 = validate(val_loader,model, comb_embeddingv,labelsv, type ='single')
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             epoch_for_best_f1 = file
    #         print(f'File {epoch_for_best_f1} gives {best_f1}')
    #     else:

    #         f1,f_we = validate(val_loader,model, comb_embeddingv,labelsv, type ='multi')
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             epoch_for_best_f1 = file
    #         print(f'File {epoch_for_best_f1} gives {best_f1}')

    

    
    model.load_state_dict(torch.load(args.weight))
    

    if args.type == 'single':

        f1 = validate(val_loader,model, comb_embeddingv,labelsv, type ='single')
        
        print(f'File {args.weight} gives {f1}')
    else:

        f1,f_we = validate(val_loader,model, comb_embeddingv,labelsv, type ='multi')
        print(f'File {args.weight} gives macro {f1} abnd weighted as {f_we}')
        
        
        

    



