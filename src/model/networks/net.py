import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List


class Network(nn.Module):
    '''
    Abstract class to be inherited by the various critic and actor classes.
    '''
    def __init__(self,
                input_shape: Tuple,
                layer_neurons: int,
                network_name: str,
                checkpoint_directory_networks: str,
                device: str = 'cpu',
                ) -> None :
        """
        Args:
            input_shape (tuple) : state space의 차원 수 
            layer_neurons (int) : 네트워크 레이어 안의 뉴런 수 ###
            name (str): 네트워크 이름
            checkpoint_directory_networks (str = 'saved_networks'): base directory for the checkpoints
        
        Returns:
            no value
        """
        super(Network, self).__init__()
        self.network_name = network_name
        self.checkpoint_directory_networks = checkpoint_directory_networks
        base_dir = checkpoint_directory_networks.split("/saved_outputs")[0]
        self.checkpoint_file_network = os.path.join(self.checkpoint_directory_networks, f'{self.network_name}.pth')
        self.checkpoint_directory_pretrain = base_dir+"/bestmodel/weights"

        self.device = device
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        self.loss_term = 1

        # if torch.cuda.device_count() > 1:
        #     # gpu 병렬처리 해라
        #     self = torch.nn.DataParallel(self)
        # self.to(self.device)

    def forward(self, 
                *args, 
                **kwargs, ) -> torch.tensor:
        raise NotImplementedError
    
    def save_network_weights(self) -> None:
        '''Save checkpoint, used in training mode'''
        torch.save(self.state_dict(), self.checkpoint_file_network) ##

    def load_network_weights(self) -> None:
        '''Save checkpoint, used in test mode'''
        self.load_state_dict(torch.load(self.checkpoint_file_network, map_location=self.device))





class Resblock(nn.Module):
    def __init__(self, cnn_params:dict) -> None:
        super(Resblock, self).__init__()
        self.cnn1 = nn.Sequential(
                            nn.Conv2d(in_channels = cnn_params["out_c"],
                              out_channels = cnn_params["out_c"],
                              kernel_size = cnn_params["kernel"],
                              stride = cnn_params["stride"],
                              padding = cnn_params["padding"]),
                            nn.BatchNorm2d(num_features=cnn_params["out_c"]),
                            nn.ReLU(),
                            nn.Conv2d(in_channels = cnn_params["out_c"],
                              out_channels = cnn_params["out_c"],
                              kernel_size = cnn_params["kernel"],
                              stride = cnn_params["stride"],
                              padding = cnn_params["padding"]),
                              nn.BatchNorm2d(num_features=cnn_params['out_c']),
                                       )
        self.relu = nn.ReLU()

    def forward(self, x):
        # resnet
        x_shortcut = x
        x = self.cnn1(x) #unsqueeze(dim=1)
        x = self.relu(torch.add(x, x_shortcut)) # out 확인
        return x







class DynamicEncoder(nn.Module):
    def __init__(self,
            input_shape,
            network_name:str = "dynamic_enc",
            dim:int = -1,
            ) -> None:
        super(DynamicEncoder, self).__init__()
        '''
        # Feature-informativeness attention encoder
            * Same effect but more powerful & easier
            * Reference : https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Multimodal_Dynamics_Dynamical_Fusion_for_Trustworthy_Multimodal_Classification_CVPR_2022_paper.pdf
        
        - n_feature(int) : the input number of the dimension to compute attention
        - dim(int) : 어텐션을 적용한 디멘젼

        '''
        self.input_shape = input_shape
        self.dim = dim 
        self.sigmoid = nn.Sigmoid()

        if type(input_shape) == int:
            self.n_feat = input_shape
        else:
            self.n_feat = input_shape[self.dim] # same with M in paper

        self.w = nn.Linear(self.n_feat, self.n_feat)  #asset_num


    
    def _encoder(self, x):
        # calculte the dynamic encoder
        self.wm = self.sigmoid(self.w(x)) # same with x.shape
        out = self.wm*x # same with x.shape, elementwise mul # #####

        # loss term for Dynamic Encoder
        l1_norm = torch.linalg.norm(self.wm, ord=1, dim=1) # (b, n_feats)
        sparsity_loss = torch.mean(l1_norm, dim=-1)  # (b) # 이건 전체 디멘션으로 할지 고민
        return out, sparsity_loss
    
        
    def forward(self, x):
        '''
        Input
        - x (B, assets=30, features=5 or 3)
        Output
        - out(B, assets, n_feats) : same effect with attentioned output
        - sparcity_loss(B, ) : loss term for dynamic encoder loss
        - attention (B, assets, n_feats)
        '''
        #scenario : 에셋별로도 어텐션 하고 싶을 수도 있지 않니? ㅠㅠ
        # assert x.dim()==3, f'x.dim() should be 3. But ({x.dim()})'
        
        if self.dim != -1 or x.shape[self.dim] != x.shape[-1]:
            # if dimension이 특정값인 경우
            x = x.transpose(self.dim,-1).contiguous()
            # calculte the dynamic encoder
            out, sparsity_loss = self._encoder(x)
            # recover the dimension of data
            out = out.transpose(self.dim, -1)
        
        else:
            # calculte the dynamic encoder
            out, sparsity_loss = self._encoder(x)
            
        return out, sparsity_loss
    








    




class Attention(nn.Module):
    def __init__(self, enc_dim:int, hid_dim:int, att_type:str='scaled'):
        '''
        input
        - enc_dim(int) : attention을 적용할 디멘션 수 
        - hid_dim(int) : hidden state's dimension
        - type(str, scaled) : [scaled, dot, badanau]
            - scaled : scaled_dot attention
            - dot : dot-product attention
            - bahdanau 
        '''
        super().__init__()
        # model option
        if att_type not in ['scaled', 'dot', 'bahdanau']:
            print(f"attention type should be [scaled, dot, bahdanau], but {att_type}. ")
            print("Set the default attention module type : scaled-dot ")
            att_type = 'scaled'
        
        self.type = att_type
        self.enc_dim = enc_dim
        self.hid_dim = hid_dim

        # model neural layers
        self.query_layer = nn.Linear(enc_dim, hid_dim)
        self.key_layer = nn.Linear(enc_dim, hid_dim)
        self.value_layer = nn.Linear(enc_dim, enc_dim)
        self.tanh = nn.Tanh()


        if self.type == 'dot':
            self.attention = self._dot_att
        elif self.type == 'badanau':
            self.bahdanau_att = nn.Linear(hid_dim, enc_dim)
            self.attention = self._bahdanau_att
        else : 
            # self.type == 'scaled'
            self.att_net = nn.Sequential()
            self.attention = self._scaled_att



    def _scaled_att(self, query:torch.tensor, key:torch.tensor):
        '''
        Calculate Scaled dot-product Attention
        https://velog.io/@cha-suyeon/%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%93%9C-%EB%8B%B7-%ED%94%84%EB%A1%9C%EB%8D%95%ED%8A%B8-%EC%96%B4%ED%85%90%EC%85%98Scaled-dot-product-Attention

        input
        - query(torch.tensor, [B, Factor, Assets])  
        - key(torch.tensor, [B, Factor, Assets])
        - d(int, 32): a scale hyperparameter, would be 2^n

        attributes?
        - ?
        - att_w (torch.tensor, [B, 1, Factor]) : attention weight

        output
        - att_w
        '''
        d = query.shape[1]
        # 차원 맞는지 확인
        qk_dot = torch.matmul(query, key.transpose(1, 2))/np.sqrt(d) # (B,F,A)*(B,A,F) -> (B, F, F)
        # masking 추가? 고민해보기
        att_w = F.softmax(qk_dot, dim=-1) # torch.matmul(att_w, value)
        return att_w
    

    def _dot_att(self, query:torch.tensor, key:torch.tensor):
        '''
        Calculate dot-product Attention

        Input
        - query(torch.tensor, [B, Factor, Assets])  
        - key(torch.tensor, [B, Factor, Assets])

        Output
        - att_w (torch.tensor, [B, 1, Factor]) : attention weight
        '''
        att_w = torch.matmul(query, key.transpose(1, 2)) # (B,F,A)*(B,A,F) -> (B,F,F) 
        return att_w # torch.matmul(att_w, value)
    

    def _bahdanau_att(self, query:torch.tensor, key:torch.tensor):
        '''
        Calculate dot-product Attention

        Input
        - query(torch.tensor, [B, Factor, Hidden])  
        - key(torch.tensor, [B, Factor, Hidden])
        - value(torch.tensor, [B, Factor, Assets])

        Output
        - att_w (torch.tensor, [B, 1, Factor]) : attention weight
        '''
        x = self.tanh(torch.add(query, key)) # (B, F, Hidden)
        align = self.bahdanau_att(x) # (B, F, Hidden) # v.transpose = self.bahdanau_att -> (B,F,F) 
        att_w = F.softmax(align, dim=-1) # (B,F,F) or (B, 1, F)
        return att_w # torch.matmul(att_w, value)
    
        
    def forward(self, x):
        '''
        input
        - x(torch.tensor, [B, Asset, Factor])  
        '''
        assert x.dim() == 3, f'x.dim() should be 3, but {x.dim()}'
        
        # change the dimension of x
        x = x.transpose(1, 2) # [B, Asset, Factor] -> [B, Factor, Asset]

        # calculate q,k,v
        query = self.query_layer(x) # [B, Factor, Hidden]
        key = self.key_layer(x) # [B, Factor, Hidden]
        value = self.value_layer(x) # [B, Factor, Asset]

        self.wm = self.attention(query=query, key=key)
        out = self.wm*value # # [B, Factor, Asset]

        # because of DynamicEncoder..
        return out, None
    
    



'''
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        self.batch_size = encoder_outputs.shape[1] 
        src_len = encoder_outputs.shape[0]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch size, src len, dec hid dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) # [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2) # [batch size, src len]
        
        return F.softmax(attention, dim=1)
'''