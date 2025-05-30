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
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    criterion = nn.BCELoss()
    valid_loss = 0
    user = 306
    for batch in enumerate(loader):

        b              = batch[0]
        users          = batch[1][1][:true_labels.shape[0]] 
        product        = batch[1][1][true_labels.shape[0]:]
        edge_ind       = batch[1][2][0]
        edge_emb_index = batch[1][2][1]
        

        
        batch_user_emb = comb_embeddingv[users]
        
        edge_features  = eev[edge_emb_index]

        batch_prod_emb = comb_embeddingv[product-6]
        
        # ----------------------------------------------------------------------
        batched_states = batch_user_emb, edge_ind, edge_features, batch_prod_emb
        #-----------------------------------------------------------------------

            
        
        
        
        predictions = model(batched_states)  # eval mode
        valid_loss = criterion(predictions.cuda(), torch.tensor(true_labels, dtype=torch.float).cuda())
        # pdb.set_trace()
        model.eval()
                
        
        
        # #pdb.set_trace()
        if type == 'single':
            predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='micro') 
            print(f'Accuracy is {acc} \nF1 score is {f1}')
        else:
            y_true = []
            y_pred = []



            predictions = predictions.round()
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
            


    return valid_loss
            
    




import torch

def swap_rows(tensor):
    
  new_tensor = tensor.clone()
  new_tensor[0, :] = tensor[1, :]
  new_tensor[1, :] = tensor[0, :]
  return new_tensor







        
        







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
        '--log',
        help='lcoation of the weights to be saved',
        type=str,
        default=1000
    )   
   

    parser.add_argument(
        '--batch_size',
        help='batch size',
        type=int,
        default=1000
    )
   
    
    import pdb,copy
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    
    
    lr = 0.001
    patience = 7
    early_stopping = 30
    num_workers = 1
    epochs = 2000
    # lambda_l = args.lambda_l
    # lambda_att = args.lambda_att
    import os
    if not os.path.exists('logs/' + args.log):
        os.makedirs('logs/' + args.log)

    os.system('cp train_multi_label.py ' +'logs/' + args.log +'/train_multi_label.py')
    # os.system('cp models/GAT_baseline_torch/model_corr.py ' +'logs/' + args.log +'/model_corr.py')
    os.system('cp model_corr.py ' +'logs/' + args.log +'/model_corr.py')
    


    

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
    
    from model import *
    user = 1000
    
    model = GATPolicy().cuda()
    
    # model.to(device)
    


    ##;
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)
    scheduler  = CosineAnnealingLR(optimizer, epochs, eta_min=lr)

 

    #----------------------------------------------------------------------------------------------------
    

    ue       = nn.Parameter(torch.empty(1000,12))
    nn.init.xavier_uniform_(ue)
    pe       = torch.tensor(np.load(args.data +'train/pe.npy'), dtype=torch.double )
    # ee       = torch.tensor(np.load(args.data +'Train/ee.npy'), dtype=torch.double )
    ei       = np.load(args.data +'train/ei.npy').T   #2,100
    ee       = nn.Parameter(torch.empty(ei.shape[1],768))



    zeros_array = torch.zeros((1000, 3551-12))
    ue = torch.hstack((ue, torch.tensor(zeros_array)))
    # pdb.set_trace()

    if np.min(ei[1])==0:
        ei[1] += (np.max(ei[0])+1)


   




    
    from torch_geometric.utils import to_undirected
    ei = torch.tensor(ei, dtype= torch.long)
    ei, ee = to_undirected(ei, ee)
    labels = np.load(args.data + 'train/lab.npy')
    # pdb.set_trace()


    comb_embedding = torch.tensor( torch.cat( (ue, pe), dim = 0 ) )
    #-------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------
    

    uev       = nn.Parameter(torch.empty(306,12))
    nn.init.xavier_uniform_(uev)

    pev       = torch.tensor(np.load(args.data +'test/pe.npy'), dtype=torch.double )
    
    eiv       = np.load(args.data +'test/ei.npy').T   #2,100
    eev       = nn.Parameter(torch.empty(eiv.shape[1],768))

    zeros_array = torch.zeros((306, 3551-12))
    uev = torch.hstack((uev, torch.tensor(zeros_array)))




    if np.min(eiv[1])==0:
        eiv[1] += user


    
    from torch_geometric.utils import to_undirected
    eiv = torch.tensor(eiv, dtype= torch.long)
    eiv, eev = to_undirected(eiv, eev)
    labelsv = np.load(args.data + 'test/lab.npy')
    # ##pdb.set_trace()


    comb_embeddingv = torch.tensor( torch.cat( (uev, pev), dim = 0 ) )
    #----------------------------------------------------------------------------------------------------------









    

    

    data = Data(x=comb_embedding, ei=ei.long(), edge_attr= (ee),y= (labels))
    datav = Data(x=comb_embeddingv, ei=eiv.long(), edge_attr= (eev),y= (labelsv))
    # data2 = Data(x=comb_embedding, ei=ei.long(), edge_attr= (ee),y= (labels))
    

  


    l1 = [i for i in range(ue.shape[0])]       
    train_mask = torch.tensor(l1,dtype=torch.long)
    data.train_idx =  train_mask
    train_loader = NeighborSampler(ei, [-1], node_idx=train_mask,
                                 batch_size=args.batch_size, shuffle=False,
                                num_workers=10)




    l2 = [i for i in range(uev.shape[0])]       
    val_mask = torch.tensor(l2,dtype=torch.long)
    datav.train_idx =  val_mask
    val_loader = NeighborSampler(eiv, [-1], node_idx=val_mask,
                                 batch_size=len(val_mask), shuffle=False,
                                num_workers=10)



    num_neighbors = [-1] 
    user =1000

    shuffle = False            
    replace = False   
    criterion = nn.BCELoss(reduction='mean')
    

    for epoch in range(2000):
        print("epoch ", epoch)
        train_loss = 0
        best_loss = np.inf
        plateau_count = 0

        k = 0

        for batch in enumerate(train_loader):
            
            
            
          
            k+=1
            b              = batch[0]
            users          = batch[1][1][:args.batch_size] 
            product        = batch[1][1][args.batch_size:]
            # persona      = 
            edge_ind       = batch[1][2][0]
            edge_emb_index = batch[1][2][1]
            # pdb.set_trace()
            

            label = labels[b*args.batch_size:(b+1)*args.batch_size][:,:].astype(np.float32)
            batch_user_emb = comb_embedding[users]
            
            
            edge_features  = ee[edge_emb_index]
            # pdb.set_trace() 
            batch_prod_emb = comb_embedding[product-6]
            # pdb.set_trace()
            # ----------------------------------------------------------------------
            batched_states = batch_user_emb, edge_ind, edge_features, batch_prod_emb
            #-----------------------------------------------------------------------

            
            optimizer.zero_grad()
            predictions = model(batched_states)  # eval mode
            # pdb.set_trace()
            
            loss = criterion(predictions.cuda(), torch.tensor(label).cuda())
            # print(loss)
            loss.backward()
            optimizer.step()
            #----------------------
            scheduler.step(loss)
            #----------------------
            train_loss += loss.item()
        print(f'Training loss for epoch {epoch} : {train_loss}')
        


        #--------------------------------------------------------------------------------
        if epoch%3 ==0:
            valid_loss = validate(val_loader,model, comb_embeddingv,labelsv, type ='multi')
            
            model_dir= os.getcwd() +"/logs/"+args.log +'/'
            model.save_state(os.path.join(model_dir, 'params_at_epoch{}.pth'.format(epoch)))
            model.save_state(os.path.join(model_dir, 'checkpoint.pth'))

        #--------------------------------------------------------------------------------
 




  
    # from torch_geometric.utils import to_networkx
    # import networkx as nx
    # G=to_networkx(data)
    # c = nx.strongly_connected_components(G)
    # print(len([k for k in c]))
    # from networkx.algorithms import bipartite

    # # Assuming you have a bipartite graph G with node attributes 'bipartite_0' and 'bipartite_1'

    # top_nodes, bottom_nodes = bipartite.sets(G)

    # pos = bipartite_layout(G, top_nodes, align='vertical')

    # plt.figure(figsize=(12, 8))
    # nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_color='lightblue', node_size=10, alpha=0.5)
    # nx.draw_networkx_nodes(G, pos, nodelist=bottom_nodes, node_color='lightgreen', node_size=10, alpha=0.5)
    # nx.draw_networkx_edges(G, pos, alpha=0.1)
    # plt.axis('off')
    # plt.savefig("abc.png")