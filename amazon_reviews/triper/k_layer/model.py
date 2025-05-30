import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter, Linear

#from weights_init import glorot, zeros
from torch_geometric.utils import softmax
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_geometric.nn import GATConv,GCNConv,SAGEConv
import pdb
class PreNormException(Exception):
    pass

class BaseModel(torch.nn.Module):
    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                self.initializer(l.weight.data)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train(self, state):
        with torch.no_grad():
            try:
                self.forward(state)
                return False
            except PreNormException:
                return True

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) \
                    and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))



class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation, heads=2, center_left=True,):
        super().__init__()
        # print('heads', heads)
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, edge_dim = out_channels, add_self_loops=False, concat=True)#, fill_value =1.0)#add_self_loops =False)
        #self.conv1 = SAGEConv(in_channels, out_channels, )GATill_value =1.0)#add_self_loops =False)
        self.linear_transform = nn.Linear(out_channels*heads, out_channels)

        self.activation= activation

        self.feature_module_final = nn.Sequential(
            # PreNormLayer(1, shift=False),  # normalize after summation trick
            self.activation,
            nn.Linear(out_channels, out_channels, bias=True)
        )

        self.post_concat = torch.nn.Linear(out_channels*2, out_features=out_channels)   
    
        
        
        
        

    def forward(self, inputs):
        left_features, edge_index, edge_features, right_features = inputs           # actuall right featuers are getting used
        
        
        x=(left_features,right_features)
        
        x, _ = self.conv1(x, edge_index, edge_attr=edge_features, return_attention_weights =True)
        x = x.relu()
        # ###pdb.set_trace()trace()
        x = self.linear_transform(x)#.relu()
        x = self.feature_module_final(x)
        
        
        
        left_features_updated = torch.cat((right_features,x), dim=-1)   
        # left_features_updated = torch.cat((left_features,x), dim=0)
        
        left_features_updated=self.post_concat(left_features_updated)

        return left_features_updated, _

class GATPolicy(BaseModel):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self, k_layers=3):
        super(GATPolicy, self).__init__()

        self.emb_size = 768
        self.k_layers = k_layers
        self.user_nfeats = 3551
        self.prod_nfeats = 3551
        self.pers_nfeats = 3551
        # self.cons_nfeats = 5
        self.edge_nfeats = 8
        # self.var_nfeats = 19
        
        # print('self.emb_size',self.emb_size)

        self.activation = nn.ReLU()
        self.initializer = lambda x: torch.nn.init.orthogonal_(x, gain=1)
        self.dict_norm_task = None

        # CONSTRAINT EMBEDDING
        # transforms the 1* 5 into 1*32
        # self.cons_embedding = nn.Sequential(
        #     nn.Linear(self.cons_nfeats, self.emb_size, bias=True),
        #     self.activation,
        #     nn.Linear(self.emb_size, self.emb_size, bias=True),
        #     self.activation
        # )
        #
        # EDGE EMBEDDING
        # similarly 1*1 to 1*32
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.edge_nfeats, self.emb_size, bias=True),
        )

        self.user_embedding = nn.Sequential(
            nn.Linear(self.user_nfeats, self.emb_size, bias=True),
        )

        self.prod_embedding = nn.Sequential(
            nn.Linear(self.prod_nfeats, self.emb_size, bias=True),
        )

        self.pers_embedding = nn.Sequential(
            nn.Linear(self.pers_nfeats, self.emb_size, bias=True),
        )
        #
        # # VARIABLE_EMBEDDING

        #simialry for variables
        # self.var_embedding = nn.Sequential(
        #     nn.Linear(self.var_nfeats, self.emb_size, bias=True),
        #     self.activation,
        #     nn.Linear(self.emb_size, self.emb_size, bias=True),
        #     self.activation
        # )

        # GRAPH CONVOLUTIONS
        # there are 2 convolutions  2 GATs (l to r)  &  (r to l)
        self.conv_v_to_c_layers = nn.ModuleList()
        self.conv_c_to_v_layers = nn.ModuleList()
        self.conv_pro_to_per = nn.ModuleList()
        
        print('Number of layers ', k_layers)
        for _ in range(self.k_layers):
            # For Product -> User updates
            self.conv_v_to_c_layers.append(
                    GATNet(self.emb_size, self.emb_size, self.emb_size, activation=self.activation))
             # For User -> Product updates
            self.conv_c_to_v_layers.append(
                    GATNet(self.emb_size, self.emb_size, self.emb_size, activation=self.activation))
         
            self.conv_pro_to_per.append(
                    GATNet(self.emb_size, self.emb_size, self.emb_size, activation=self.activation))
         


        self.initialize_parameters()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc0 = nn.Linear(768*2, 768)
        self.bn0 = nn.BatchNorm1d(768)
        self.fc1 = nn.Linear(768, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        # self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = torch.max(n_vars_per_sample)

        output = torch.split(
            tensor=output,
            split_size_or_sections=n_vars_per_sample.tolist(),
            dim=1,
        )

        output = torch.cat([
            F.pad(x,
                pad=[0, n_vars_max - x.shape[1], 0, 0],
                mode='constant',
                value=pad_value)
            for x in output
        ], dim=0)

        return output


 

    

    def cartesian_product(self, A, B):
        """
        Computes the Cartesian product of two matrices.

        Args:
        A: A matrix of shape (m, n).
        B: A matrix of shape (p, n).

        Returns:
        A matrix of shape (m * p, 2 * n) representing the Cartesian product.
        """

        # Repeat each row of A 'p' times (number of rows in B)
        repeated_B = B.repeat(A.shape[0], 1)

        # Tile B 'm' times (number of rows in A)
        tiled_A = A.repeat_interleave(B.shape[0], dim=0)

        # Concatenate the repeated A and tiled B matrices
        C = torch.cat((repeated_B, tiled_A), dim=1)

        return C
        


    def forward(self, inputs, returns=None):
        
        

        user_features, edge_indices, edge_features, product_features, persona_features, persona_prod_edge_ind = inputs
        # pdb.set_trace()
        
        min_val = edge_indices[0].min()
        # ###pdb.set_trace()trace()                         # gat always takes the graph with all the edgeindex starting from 0 ordere irrepsecitve of the ordera and the index they have in the graph
                                                            # thus the subtraction of the m
        edge_indices[0] -= min_val
        # edge_indices = torch.zeros(2,0).int()
        edge_features = torch.zeros(edge_features.shape[0],768).cuda()
        ##pdb.set_trace()trace()
        # edge_features = self.edge_embedding(edge_features)

        user_features    = self.user_embedding(user_features.float()).cuda()
        product_features = self.prod_embedding(product_features.float()).cuda()
        persona_features = self.pers_embedding(persona_features.float()).cuda()


        user_features = user_features
        edge_indices = edge_indices
        edge_features = edge_features
        product_features = product_features

        edge_indices_rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0).cuda()

        min_val1 = persona_prod_edge_ind[1].min()
        persona_prod_edge_ind[1] -= min_val1
        ##pdb.set_trace()trace()
        edge_features_per = torch.zeros(persona_prod_edge_ind.shape[1],768).cuda()



        for i in range(self.k_layers):
             # 1. Update User Features (Product -> User)
            # Ensure you have lists: self.conv_v_to_c_layers and self.conv_c_to_v_layers in __init__

             # 2. Update Product Features (User -> Product)
            product_features_new, _ = self.conv_c_to_v_layers[i]((
                user_features, edge_indices_rev, edge_features, product_features
            ))
            product_features = self.activation(product_features_new) # Apply activation
             # Add residual connection, dropout, batchnorm etc. here if needed
            
            user_features_new, _ = self.conv_v_to_c_layers[i]((
                product_features, edge_indices, edge_features, user_features
            ))
            user_features = self.activation(user_features_new) # Apply activation
            # Add residual connection, dropout, batchnorm etc. here if needed

            # --- Persona Feature Update ---
            persona_features_new, _ = self.conv_pro_to_per[i](( # Or use a layer from a list if multiple persona layers exist
                product_features, persona_prod_edge_ind, edge_features_per, persona_features
            ))
            persona_features = self.activation(persona_features_new)
         





        ##pdb.set_trace()trace()

        
        all_pairs = self.cartesian_product(user_features, persona_features).cuda()
        
        # all_pairs = all_pairs.view(-1, 2 * 768)


        x = self.relu(self.bn0(self.fc0(all_pairs)))
        x = self.dropout(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.sigmoid(self.fc3(x))
        # pdb.set_trace()
        return x, user_features, product_features





        

        

        # return output
        
        


