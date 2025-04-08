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



def validate(args, model, type = 'single', test=False):
    criterion = nn.BCELoss()
    valid_loss = 0
    #----------------------------------------------------------------------------------------------------
    

    ue           = torch.tensor(np.load(args.data +'val/ue.npy'), dtype=torch.float32 ).cuda()
    pe           = torch.tensor(np.load(args.data +'val/pe.npy'), dtype=torch.float32 ).cuda()
    ee           = torch.tensor(np.load(args.data +'val/ee.npy'), dtype=torch.float32 ).cuda()
    pers_emb     = torch.tensor(np.load(args.data +'val/pers_emb.npy'), dtype=torch.float32 ).cuda()


    pers_emb = nn.Parameter(torch.empty(9, 1879))
    nn.init.xavier_uniform_(pers_emb)
    

    zeros_array = np.zeros((2000, 1867))
    ue = torch.hstack((ue, torch.tensor(zeros_array).cuda()))

    

    # edgeindex are always in the form of y to x in the (2,n) matrix
    ei_per2pro   = swap_rows(torch.tensor(np.load(args.data +f'val/edge_pro2per_allEdgeWithGreatorThanCount_0.npy'), dtype=torch.long ).T).cuda() #y,x fornat
    # ei_per2pro   = swap_rows(torch.tensor(np.load(args.data +f'val/edge_pro2per_edgeCountThreshold_{args.threshold}.npy'), dtype=torch.long ).T).cuda() #y,x fornat
    ei_pro2u     = (torch.tensor(np.load(args.data +'val/ei_pro2u.npy'), dtype=torch.long ).T).cuda()  #y,x fornat
    ei_u2pro     = (torch.tensor(np.load(args.data +'val/ei_u2pro.npy'), dtype=torch.long ).T).cuda()  #y,x fornat

    
    # ei_undirected_user_pro = torch.cat((ei_pro2u, ei_u2pro), axis = 1 )
    from torch_geometric.utils import to_undirected
    ei_undirected_user_pro, _ = to_undirected(ei_u2pro, ee)
    # ei_undirected_user_pro_per = torch.cat((ei_undirected_user_pro, ei_pro2per), axis = 1 )
    ei_undirected_user_pro_per = torch.cat((ei_pro2u, ei_u2pro), axis = 1 )
    #
    labels = torch.tensor(np.load(args.data + 'val/lab.npy'))


    user =2000
    per = 9
    comb_embedding = torch.tensor( torch.cat( (pers_emb.cuda(), ue.cuda(), pe.cuda() ), dim = 0 ) ).cuda()
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
            
        print("epoch ", epoch)
        
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

        #ablation to check the effectof persona embeddings initialization
        # pdb.set_trace()
        torch.nn.init.xavier_uniform_(persona_embeddings)  
        
        torch.nn.init.xavier_uniform_(persona_embeddings)  
        # #pdb.set_trace()
        

        

        

        # label = labels[b*args.batch_size:(b+1)*args.batch_size][:,:].astype(np.float32)
        batch_user_emb = comb_embedding[users].cuda()
        
        edge_features  = torch.zeros(edge_ind.shape[1], 768).cuda()

        batch_prod_emb = comb_embedding[product].cuda()
        # #pdb.set_trace()
        
        # ----------------------------------------------------------------------
        batched_states = batch_user_emb.cuda(), edge_ind.cuda(), edge_features.cuda(), batch_prod_emb.cuda(), persona_embeddings.cuda(), persona_prod_edge_ind.cuda()
        #-----------------------------------------------------------------------

        
        
        predictions,_,_,_ = model(batched_states)  # eval mode

        permuted_labels = permute_labels(labels.cuda(), persona_node.cuda())
        labels_ = permuted_labels.reshape(-1,persona_embeddings.shape[0]*batch_user_emb.shape[0])
        # #pdb.set_trace()

        predictions = predictions.squeeze().cuda()
        labels_     = (labels_).squeeze().cuda()
        
        
        loss = criterion(predictions.cuda(), torch.tensor(labels_).squeeze().type(torch.float32).cuda())
        # pdb.set_trace()

        predictions = predictions.detach().cpu().numpy()
        labels_     = labels_.detach().cpu().numpy()
            
        # print(f'Validation {epoch} : {train_loss}')
        # #
        
                
        
        
        # ##
        if type == 'single':
            predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='micro') 
            print(f'Accuracy is {acc} \nF1 score is {f1}')
        else:
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
        default=8
    )

    parser.add_argument(
        '--threshold',
        help='batch size',
        type=int,
        default=1
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

    os.system('cp train_multi_label_dummy.py ' +'logs/' + args.log +'/train_multi_label_dummy.py')
    # os.system('cp models/GAT_baseline_torch/model_corr.py ' +'logs/' + args.log +'/model_corr.py')
    os.system('cp model_corr_multi_dummy.py ' +'logs/' + args.log +'/model_corr_multi_dummy.py')
    


    

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
    
    model = GATPolicy()
    model.cuda()
    


    ##;
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)
    scheduler   = CosineAnnealingLR(optimizer, epochs, eta_min=lr)

 

    #----------------------------------------------------------------------------------------------------
    

    ue           = torch.tensor(np.load(args.data +'train/ue.npy'), dtype=torch.float32 ).cuda()
    pe           = torch.tensor(np.load(args.data +'train/pe.npy'), dtype=torch.float32 ).cuda()
    ee           = torch.tensor(np.load(args.data +'train/ee.npy'), dtype=torch.float32 ).cuda()
    pers_emb     = torch.tensor(np.load(args.data +'train/pers_emb.npy'), dtype=torch.float32 ).cuda()



    pers_emb = nn.Parameter(torch.empty(9, 1879))
    nn.init.xavier_uniform_(pers_emb)



    zeros_array = np.zeros((7000, 1867))
    ue = torch.hstack((ue, torch.tensor(zeros_array).cuda()))
    # pdb.set_trace()



    
    # edgeindex are always in the form of y to x in the (2,n) matrix
    # ei_per2pro   = swap_rows(torch.tensor(np.load(args.data +f'train/edge_pro2per_edgeCountThreshold_{args.threshold}.npy'), dtype=torch.long ).T).cuda() #y,x fornat
    # pdb.set_trace()
    ei_per2pro   = swap_rows(torch.tensor(np.load(args.data +f'train/edge_pro2per_allEdgeWithGreatorThanCount_0.npy'), dtype=torch.long ).T).cuda() #y,x fornat
    
    ei_pro2u     = (torch.tensor(np.load(args.data +'train/ei_pro2u.npy'), dtype=torch.long ).T).cuda()  #y,x fornat
    ei_u2pro     = (torch.tensor(np.load(args.data +'train/ei_u2pro.npy'), dtype=torch.long ).T).cuda()  #y,x fornat

    
    # ei_undirected_user_pro = torch.cat((ei_pro2u, ei_u2pro), axis = 1 )
    from torch_geometric.utils import to_undirected
    ei_undirected_user_pro, _ = to_undirected(ei_u2pro, ee)
    # ei_undirected_user_pro_per = torch.cat((ei_undirected_user_pro, ei_pro2per), axis = 1 )
    ei_undirected_user_pro_per = torch.cat((ei_pro2u, ei_u2pro), axis = 1 )
    #


    


    labels = torch.tensor(np.load(args.data + 'train/lab.npy'))



    comb_embedding = torch.tensor( torch.cat( (pers_emb.cuda(), ue.cuda(), pe.cuda() ), dim = 0 ) ).cuda()
    comb_embedding = comb_embedding.cuda()
    #-------------------------------------------------------------------------------------------------------

    

    
    ee = torch.zeros(ei_undirected_user_pro.shape[0],768).cuda()
    data = Data(x=comb_embedding, ei=ei_undirected_user_pro_per.long(), edge_attr= (ee),y= (labels))
    # datav = Data(x=comb_embeddingv, ei=eiv.long(), edge_attr= (eev),y= (labelsv))
    # data2 = Data(x=comb_embedding, ei=ei.long(), edge_attr= (ee),y= (labels))
    

  


    l1 = [i for i in range(9,ue.shape[0]+9)]     
    train_mask = torch.tensor(l1,dtype=torch.long).cuda()
    # data.train_idx =  train_mask
    train_loader = NeighborSampler(ei_undirected_user_pro_per, [-1], node_idx=train_mask,
                                 batch_size=ue.shape[0], shuffle=False,
                                num_workers=1)                 #neighbour sampler  -  always reorders the nodes of whose netighiots it has to be find from 0 to whaever. - however it lets the indexing of the 
                                                                #neighoburs remain the same 

    

    
    # l2 = [i for i in range(0,pers_emb.shape[0])]       
    # train_mask1 = torch.tensor(l2,dtype=torch.long)
    # data1.train_idx =  train_mask1                                                                


    # train_loader_persona = NeighborSampler(ei_undirected_user_pro_per, [-1], node_idx=train_mask1,
    #                              batch_size=pers_emb.shape[0], shuffle=False)                 #neighbour sampler  -  

    # data_iter_persona = iter(train_loader_persona)



    pers = 9
    user = 7000



    # l2 = [i for i in range(uev.shape[0])]       
    # val_mask = torch.tensor(l2,dtype=torch.long)
    # datav.train_idx =  val_mask
    # val_loader = NeighborSampler(eiv, [-1], node_idx=val_mask,
    #                              batch_size=len(val_mask), shuffle=False,
    #                             num_workers=10)



    # num_neighbors = [-1] 

    # shuffle = False            
    # replace = False   
    criterion = nn.BCELoss(reduction='mean')
    

    for epoch in range(2000):
        train_loss = 0
        best_loss = np.inf
        plateau_count = 0

        k = 0

        for batch in enumerate(train_loader):
            
            
            
            print("epoch ", epoch)
            k+=1
            b              = batch[0]
            
            users          = batch[1][1][:user] 
            product        = batch[1][1][user:]
            
            edge_ind       = batch[1][2][0]     
            edge_emb_index = batch[1][2][1]

            batch_user_emb = comb_embedding[users].cuda()
            
            edge_features  = torch.zeros(edge_ind.shape[1], 768).cuda()

            batch_prod_emb = comb_embedding[product].cuda()
            
            

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

             #ablation to check the effectof persona embeddings initialization
            torch.nn.init.xavier_uniform_(persona_embeddings)  
            # #pdb.set_trace()
            #----------------------------------------------------------------------------------------------------------------
            

            

            

            # label = labels[b*args.batch_size:(b+1)*args.batch_size][:,:].astype(np.float32)
           
            # ----------------------------------------------------------------------
            batched_states = batch_user_emb.cuda(), edge_ind.cuda(), edge_features.cuda(), batch_prod_emb.cuda(), persona_embeddings.cuda(), persona_prod_edge_ind.cuda()
            #-----------------------------------------------------------------------

            
            optimizer.zero_grad()
            predictions, _,_,_ = model(batched_states)  # eval mode

            permuted_labels = permute_labels(labels.cuda(), persona_node.cuda())
            labels_ = permuted_labels.reshape(-1,persona_embeddings.shape[0]*batch_user_emb.shape[0])
            # #pdb.set_trace()

            predictions = predictions.squeeze().cuda()
            labels_     = torch.tensor(labels_).squeeze().cuda()
            
            
            loss = criterion(predictions.cuda(), torch.tensor(labels_).squeeze().type(torch.float32).cuda())
            # print(loss)
            import time
            start = time.time()
            loss.backward()
            optimizer.step()
            # print(time.time()-start)
            #----------------------
            scheduler.step(loss)
            #----------------------
            train_loss += loss.item()
        print(f'Training loss for epoch {epoch} : {train_loss}')
        


        #--------------------------------------------------------------------------------
        if epoch%1 ==0:
            valid_loss = validate(args ,model, type ='multi')
            
            model_dir= os.getcwd() +"/logs/"+args.log +'/'
            model.save_state(os.path.join(model_dir, 'params_at_epoch{}.pth'.format(epoch)))
            model.save_state(os.path.join(model_dir, 'checkpoint.pth'))

        # --------------------------------------------------------------------------------
 




  
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
