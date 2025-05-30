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


def swap_rows(tensor):
  

  # Create a new tensor with the rows swapped
  new_tensor = tensor.clone().cuda()
  new_tensor[0, :] = tensor[1, :]
  new_tensor[1, :] = tensor[0, :]
  return new_tensor


def permute_labels(labels, permutation_order):
    
    # Create a permutation tensor based on the desired order
    labels = torch.tensor(labels, dtype=torch.long)
    permutation_tensor = torch.tensor(permutation_order, dtype=torch.long)

    # Gather the labels according to the permutation tensor
    permuted_labels = torch.gather(labels, dim=1, index=permutation_tensor.unsqueeze(0).expand(labels.shape[0], -1))

    return permuted_labels
        





def remove_edge(array, percentage):


    # Calculate the number of rows to remove
    num_rows_to_remove = int(percentage * array.shape[0])

    # Generate random indices of rows to remove
    indices_to_remove = np.random.choice(array.shape[0], num_rows_to_remove, replace=False)

    # Create a boolean mask to select the remaining rows
    mask = np.ones(array.shape[0], dtype=bool)
    mask[indices_to_remove] = False

    # Filter the array using the mask
    reduced_array = array[mask]

    return reduced_array









def validate(args, model, type = 'single', test=False):
    criterion = nn.BCELoss()
    valid_loss = 0
    #----------------------------------------------------------------------------------------------------
    

    ue           = torch.tensor(np.load(args.data +'test/ue.npy'), dtype=torch.float32 ).cuda()
    pe           = torch.tensor(np.load(args.data +'test/pe.npy'), dtype=torch.float32 ).cuda()
    ee           = torch.tensor(np.load(args.data +'test/ee.npy'), dtype=torch.float32 ).cuda()
    pers_emb     = torch.tensor(np.load(args.data +'test/pers_emb.npy'), dtype=torch.float32 ).cuda()
    
    # edgeindex are always in the form of y to x in the (2,n) matrix
    # ei_per2pro   = swap_rows(torch.tensor(np.load(args.data +'test/edge_pro2per.npy'), dtype=torch.long ).T).cuda() #y,x fornat
    # ei_pro2u     = (torch.tensor(np.load(args.data +'test/edge_pro2u.npy'), dtype=torch.long ).T).cuda()  #y,x fornat
    # ei_u2pro     = (torch.tensor(np.load(args.data +'test/edge_u2pro.npy'), dtype=torch.long ).T).cuda()  #y,x fornat



    # edgeindex are always in the form of y to x in the (2,n) matrix
    ei_per2pro   = swap_rows(torch.tensor(remove_edge( np.load(args.data +'test/edge_pro2per.npy'), args.drop), dtype=torch.long ).T).cuda() #y,x fornat
    # pdb.set_trace()

    ei_pro2u     = remove_edge(np.load(args.data +'test/edge_pro2u.npy'), args.drop)
    ei_u2pro     = swap_rows(torch.tensor(ei_pro2u))

    ei_pro2u     = (torch.tensor(ei_pro2u, dtype=torch.long ).T).cuda()  #y,x fornat
    ei_u2pro     = (torch.tensor(ei_u2pro, dtype=torch.long ).T).cuda()  #y,x fornat




    pers_emb = nn.Parameter(torch.empty(9, 1879)).cuda()
    nn.init.xavier_uniform_(pers_emb)
    

    zeros_array = np.zeros((1000, 1867))
    ue = torch.hstack((ue, torch.tensor(zeros_array).cuda()))

    
    # ei_undirected_user_pro = torch.cat((ei_pro2u, ei_u2pro), axis = 1 )
    from torch_geometric.utils import to_undirected
    ei_undirected_user_pro, _ = to_undirected(ei_u2pro, ee)
    # ei_undirected_user_pro_per = torch.cat((ei_undirected_user_pro, ei_pro2per), axis = 1 )
    ei_undirected_user_pro_per = torch.cat((ei_pro2u, ei_u2pro), axis = 1 )
    #
    labels = torch.tensor(np.load(args.data + 'test/lab.npy'))


    user =1000
    per = 9
    pdb.set_trace()
    comb_embedding = torch.tensor( torch.cat( (pers_emb, ue, pe ), dim = 0 ) ).cuda()
    comb_embedding = comb_embedding.cuda()
    #-------------------------------------------------------------------------------------------------------


    ee = torch.zeros(ei_undirected_user_pro.shape[0],768).cuda()
    data = Data(x=comb_embedding, ei=ei_undirected_user_pro_per.long(), edge_attr= (ee),y= (labels))
    
    l1 = [i for i in range(9,ue.shape[0]+9)]     
    train_mask = torch.tensor(l1,dtype=torch.long).cuda()
    # data.train_idx =  train_mask
    train_loader = NeighborSampler(ei_undirected_user_pro_per, [-1], node_idx=train_mask,
                                 batch_size=ue.shape[0], shuffle=False,
                                num_workers=1)                                                                                
    model.eval()



    for batch in enumerate(train_loader):
            
       
        
        b              = batch[0]
        
        users          = batch[1][1][:user] 
        product        = batch[1][1][user:]
        
        edge_ind       = batch[1][2][0]     #neighbot sampler gives the edge index in the reverse order and starts the indexing in continous order from 0.....something for the trainmask and 
        edge_emb_index = batch[1][2][1]
        

        #----------------------------------------------------------------------------------------------------------------
        l2 = product.cuda()
        train_mask1 = torch.tensor(l2,dtype=torch.long).cuda()

        train_loader_persona = NeighborSampler(ei_per2pro, [-1], node_idx=train_mask1,
                             batch_size = pe.shape[0], shuffle=False)                 #neighbour sampler  -  

        data_iter_persona = iter(train_loader_persona)

        #----------------------------------------------------------------------------------------------------------------


        persona = next(data_iter_persona)
        persona_node = persona[1][-9:].cuda()
        persona_neighbours = persona[1][:-9].cuda()   #this has the same order as the ones from user-prod mesage passing
        persona_prod_edge_ind = swap_rows(persona[2][0]).cuda()   # get the indexing from persona to product edges
        persona_embeddings = comb_embedding[persona_node].cuda()
        # #pdb.set_trace()
        

        

        

        # label = labels[b*args.batch_size:(b+1)*args.batch_size][:,:].astype(np.float32)
        batch_user_emb = comb_embedding[users].cuda()
        
        edge_features  = torch.zeros(edge_ind.shape[1], 768).cuda()

        batch_prod_emb = comb_embedding[product].cuda()
        # #pdb.set_trace()
        
        # ----------------------------------------------------------------------
        batched_states = batch_user_emb.cuda(), edge_ind.cuda(), edge_features.cuda(), batch_prod_emb.cuda(), persona_embeddings.cuda(), persona_prod_edge_ind.cuda()
        #-----------------------------------------------------------------------

        
        
        predictions = model(batched_states)  # eval mode

        permuted_labels = permute_labels(labels.cuda(), persona_node.cuda())
        labels_ = permuted_labels.reshape(-1,persona_embeddings.shape[0]*batch_user_emb.shape[0])
        # #pdb.set_trace()

        predictions = predictions.squeeze().cuda()
        labels_     = (labels_).squeeze().cuda()
        
        
        loss = criterion(predictions.cuda(), torch.tensor(labels_).squeeze().type(torch.float32).cuda())
        predictions = predictions.detach().cpu().numpy()
        labels_     = labels_.detach().cpu().numpy()
            
        # print(f'Validation {epoch} : {train_loss}')
        # #
        
                
        
        
        # ##
        if type == 'single' or type == 'multi':
        #     # predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        #     # accuracy = accuracy_score(true_labels, predictions)
        #     # f1 = f1_score(true_labels, predictions, average='micro') 
        #     # print(f'Accuracy is {acc} \nF1 score is {f1}')

        #     predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            
        #     f1_weighted      = f1_score(true_labels, predictions, average='micro')
        #     jaccard_weighted = jaccard_score(true_labels, predictions, average='weighted')
            
        #     hamming_loss_ = hamming_loss(true_labels, predictions) 
           
        #     print(f'Test F1 score macro: {f1_weighted}')
        #     print(f'Jaccard macro: {jaccard_weighted}')
        #     print(f'Hamming: {hamming_loss_}')

        # else:
            y_true = []
            y_pred = []



            predictions = predictions.round()
            y_true.extend(labels_)
            y_pred.extend(predictions)
            # #pdb.set_trace()
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)

            # Calculate final metrics
            ##
            
            # final_f1_micro = f1_score(y_true, y_pred, average='micro')
            # final_f1_weighted = f1_score(y_true, y_pred, average='weighted')
            # print(f'Test F1 score macro: {final_f1_micro}')
            # print(f'Test F1 score weighted: {final_f1_weighted}')
            # conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
            

            final_f1_macro      = f1_score(y_true, y_pred, average='macro')
            final_f1_weighted   = f1_score(y_true, y_pred, average='weighted')
            # final_f1_samples    = f1_score(y_true, y_pred, average='samples')     #known to be good for multilabel

            jaccard_macro    = jaccard_score(y_true, y_pred, average='macro')
            jaccard_weighted = jaccard_score(y_true, y_pred, average='weighted')
            # jaccard_samples  = jaccard_score(y_true, y_pred, average='samples')  #known to be good for multilabel

            hamming_loss_ = hamming_loss(y_true, y_pred)


            print(f'Test F1 score macro: {final_f1_macro}')
            print(f'Test F1 score weighted: {final_f1_weighted}')
            # print(f'Test F1 score samples: {jaccard_samples}')
            
            print(f'Jaccard macro: {jaccard_macro}')
            print(f'Jaccard weighted: {jaccard_weighted}')
            # print(f'Jaccard samples: {jaccard_samples}')
            
            print(f'Hamming: {hamming_loss_}')
            


    return valid_loss
    



        
        


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


    parser.add_argument(
        '--drop',
        help='',
        type=float,
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
    from model_corr_multi_dummy import *
    
    model = GATPolicy().cuda()

    
    


    
    


 
    #-------------------------------------------------------------------------------------------------------

    
    model.eval()
    model.load_state_dict(torch.load(args.weight))
    

    if args.type == 'single':

        f1 = validate(args, model, type ='single')
        
        print(f'File {args.weight} gives {f1}')
    else:

        validate(args, model,  type ='multi')
        
        
        

    

