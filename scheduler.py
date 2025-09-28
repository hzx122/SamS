import os
import time
import random
import numpy as np
from numpy.random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import numpy as np
from utils import all_gather_if_needed

from transformers import AutoTokenizer, AutoModel,AutoConfig

def block_reduce(image, block_size, func=np.sum):

    image = np.asanyarray(image)
    last_dim_size = image.shape[-1]
    new_shape = image.shape[:-1] + (last_dim_size // block_size, block_size)
    arr = image.reshape(new_shape)
    result = func(arr, axis=-1)

    return result
    

def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)
def batch_reward_mapping(raw_val,a=50,b=0.5):
    return raw_val * a + b


def cacu_bandit_reward(batch_level_reward,rm,r,p,reward_mode,beta=0.5):
    raw_sample_level_reward=rm
    if "logp" in reward_mode:
        raw_sample_level_reward=0.5*raw_sample_level_reward+0.5*(1-p)  
    if "rw" in reward_mode:
        raw_sample_level_reward=raw_sample_level_reward+r
    
    sample_level_reward=F.sigmoid(raw_sample_level_reward)

    if "add"  in reward_mode:
        batch_level_reward=batch_reward_mapping(batch_level_reward)
        final_reward=beta*F.sigmoid(torch.tensor(batch_level_reward)) + (1-beta) * F.sigmoid(sample_level_reward)
    elif "weight" in reward_mode:
        final_reward= F.sigmoid(torch.tensor(batch_level_reward))*raw_sample_level_reward*10
    return final_reward

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
      
class Residual_Network_exploitation(nn.Module):
    def __init__(self, dev, dim, hidden_size=512, k=10, num_layers=4, use_residual=False,activate="relu",norm="ln",use_dropout=False,drop_rate=0.1):
        super(Residual_Network_exploitation, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_layers=num_layers
        self.use_dropout=use_dropout
        self.input_size=dim

        self.layers.append(nn.Linear(dim, hidden_size))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, k))

        if activate=="relu":
            self.activate = nn.ReLU()
        elif activate=="silu":
            self.activate=Swish()

        if norm=="ln":
            self.layer_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        elif norm=="bn":
            self.layer_norm=nn.BatchNorm1d(num_features=hidden_size)

        
        self.dropout=nn.Dropout(p=drop_rate) 


    def forward(self, x,attention_mask):
        hiddenstates=[]
        x=self.activate(self.layers[0](x)) 
        for _ , layer in enumerate(self.layers[1:self.num_layers-1]):
            if self.use_residual:
                identity = x
                x = self.activate(layer(x)) 
                hiddenstates.append(x)
                x = x+identity 
                x=self.layer_norm(x) 
            else:
                x = self.activate(layer(x))
            
            if self.use_dropout:
                x=self.dropout(x)
        
    
        x=self.layers[-1](x) #[1,1]
        
        hiddenstates=torch.stack(hiddenstates,dim=0) 
        hiddenstates=hiddenstates.permute(1,0,2)
        hiddenstates=hiddenstates.reshape(x.shape[0],-1)

        return x,hiddenstates
    
    def init_weights_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  
            nn.init.zeros_(m.bias)
    


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Residual_Network_exploitation_with_Encoder(nn.Module):
    def __init__(self, dev, dim, hidden_size=512, k=10, num_layers=4, use_residual=False,activate="relu",norm="ln",use_dropout=False,drop_rate=0.1,connector_hidden_dim=1024,encoder_name_or_path=None,train_encoder=False):
        super(Residual_Network_exploitation_with_Encoder, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_layers=num_layers
        self.use_dropout=use_dropout
        self.train_encoder=train_encoder
        self.input_size=dim

        # construct encoder
        self.encoder_config=AutoConfig.from_pretrained(encoder_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path)
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path).encoder
        self.encoder_hidden_dim=self.encoder_config.hidden_size

        # construct connector：bridge the gap between policy and encoder
        self.connector=nn.ModuleList()
        self.connector.append(nn.Linear(dim,connector_hidden_dim)) # policy h h=1024 consistent with llava
        self.connector.append(nn.Linear(connector_hidden_dim,self.encoder_hidden_dim)) # h encoder

        self.layers.append(nn.Linear(self.encoder_hidden_dim, hidden_size))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, k))

        if activate=="relu":
            self.activate = nn.ReLU()
        elif activate=="silu":
            self.activate=Swish()

        if norm=="ln":
            self.layer_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        elif norm=="bn":
            self.layer_norm=nn.BatchNorm1d(num_features=hidden_size)

        
        self.dropout=nn.Dropout(p=drop_rate) 

    def forward(self, x,attention_mask):
        if attention_mask == None or attention_mask.shape !=x.shape[:2]:
            # print("attention mask error, replaced")
            attention_mask=torch.ones(x.shape[0],x.shape[1],dtype=torch.long).to(x.device)
        
        # print(attention_mask.shape)
        # print(x.shape)
        # connector
        x=self.activate(self.connector[0](x))  # b,s,h
        x=self.activate(self.connector[1](x))  # b,s,h

        # encoder
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False 
        
        x =self.encoder(x) # b,s,h'
        x = mean_pooling(x,attention_mask) # b,h'
        x = F.normalize(x, p=2, dim=1) 

        hiddenstates=[]

        x=self.activate(self.layers[0](x)) #[batch, hidden_dim]
        for _ , layer in enumerate(self.layers[1:self.num_layers-1]):
            if self.use_residual:
                identity = x
                x = self.activate(layer(x)) 
                hiddenstates.append(x)
                x = x+identity 
                x=self.layer_norm(x) 
            else:
                x = self.activate(layer(x))
            
            if self.use_dropout:
                x=self.dropout(x)
    
        x=self.layers[-1](x) #[1,1]
        
        hiddenstates=torch.stack(hiddenstates,dim=0) 
        hiddenstates=hiddenstates.permute(1,0,2)
        hiddenstates=hiddenstates.reshape(x.shape[0],-1)

        return x,hiddenstates
    
    def init_weights_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    

    
class Residual_Network_exploration(nn.Module):
    def __init__(self, dev, dim, hidden_size=2098, k=10, num_layers=4, block_size=4, use_residual=True,activate="relu",norm="ln",use_dropout=False,drop_rate=0.1):
        super(Residual_Network_exploration, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.block_size = block_size
        self.use_dropout = use_dropout
        self.input_size=dim

        self.layers.append(nn.Linear(dim, hidden_size))


        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, k))


        if activate=="relu":
            self.activate = nn.ReLU()
        elif activate=="silu":
            self.activate=Swish()

        if norm=="ln":
            self.layer_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        elif norm=="bn":
            self.layer_norm=nn.BatchNorm1d(num_features=hidden_size)

        self.dropout=nn.Dropout(p=drop_rate) 

    def forward(self, x):#[1792]
        h=x.clone()
        x=self.activate(self.layers[0](x)) 
        # add & norm
        x=x+h
        x=self.activate(x)

        for _ , layer in enumerate(self.layers[1:self.num_layers-1]):
            if  self.use_residual:
                # mlp
                x = self.activate(layer(x))
                # add & norm
                x = x+h
                x=self.layer_norm(x)
                # x=self.dropout(x)
            else:
                x = self.activate(layer(x))

            if self.use_dropout:
                x=self.dropout(x)

        x=self.layers[-1](x) #[1]
        return x
    
    def init_weights_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  
            nn.init.zeros_(m.bias)


def EE_forward(net1, net2, x,attention_mask): 
    net1.eval()
    net2.eval()
    with torch.no_grad():
        x=x.unsqueeze(0)
        x.requires_grad = False
        f1,hiddenstates = net1(x,None) 
        net1.zero_grad()
        hiddenstates=hiddenstates.flatten().detach()
        h = block_reduce(hiddenstates.cpu().numpy(), block_size=net2.block_size, func=np.mean) 
        h=torch.from_numpy(h).to(x.device)
        f2 = net2(h) 
    return f1, f2, h  


def batch_EE_forward(net1, net2, x,attenton_mask): 
    net1.eval()
    net2.eval()
    with torch.no_grad(): 
        x.requires_grad = False
        f1,hiddenstates = net1(x,None) 
        net1.zero_grad()
        hiddenstates=hiddenstates.detach()
        h = block_reduce(hiddenstates.cpu().numpy(), block_size=net2.block_size, func=np.mean) 
        h=torch.from_numpy(h).to(x.device)
        f2 = net2(h) 
    return f1, f2, h


def train_NN_batch(model, X, Y, 
                   num_epochs=16, 
                   lr=0.0005, 
                   batch_size=256, 
                   num_batch=4,
                   is_f1=True,
                   weight_decay=0.00001,
                   threshold=0.0005,
                   use_combine_loss=True,
                   device="cuda",
                   ref_weight=None,
                   use_encoder=False,
                   attention_mask=None
                   ): 
    model.train()

    
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    length=len(X)  
    for _ in range(num_batch):
        for _ in range(num_epochs): 
            batch_loss = 0.0
            Y=Y.unsqueeze(1).to(device)
            X=X.to(device)
            if is_f1:
                pred,_ = model(X,attention_mask)
            else:
                pred=model(X)
        
            optimizer.zero_grad()
            squared_differences = (pred - Y) ** 2
            # loss = squared_differences.sum() 
            loss=torch.mean(squared_differences)
            loss.backward()

            optimizer.step()
            
            batch_loss=loss.item()
                
            if batch_loss / length <= threshold: 
                return batch_loss / length
    return batch_loss / length 




