import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class ModelMAAR(nn.Module):
    def __init__(self, args, device,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(ModelMAAR, self).__init__()
        self.args =args

        self.drug_vocab_size = 65
        self.protein_vocab_size = 26

        self.dim = args.ds.char_dim
        self.conv = args.ds.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = args.ds.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = args.ds.protein_kernel

        self.attention_dim = args.ds.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        
        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.prot_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )

        self.Prot_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        
        #add
        self.hid_dim = 256
        self.n_heads = 8
        self.dropout_num = 0.1
        self.mhsa = SelfAttention(self.hid_dim, self.n_heads, self.dropout_num)

        self.d_c = CBlock(85, self.args.ds.c_d, 8) #16
        self.p_c = CBlock(979, self.args.ds.c_p, 8) # 256

        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Prot_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)
        
        self.prot_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)

        self.attention_layer = nn.Linear(self.conv*4,self.conv*4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, prot, drug_mask, prot_mask):

        drugembed = self.drug_embed(drug)
        protembed = self.prot_embed(prot)
        
        drugembed_cnn = drugembed.permute(0, 2, 1)
        protembed_cnn = protembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed_cnn)
        protConv = self.Prot_CNNs(protembed_cnn)
        
        #add
        drug_c = drugConv.permute(2, 0, 1)
        prot_c = protConv.permute(2, 0, 1)
        
        d_catt = self.d_c(drug_c) #att, out_finfa
        p_catt = self.p_c(prot_c)

        drug_sa = self.mhsa(drug_c, drug_c, drug_c, d_catt)
        prot_sa = self.mhsa(prot_c, prot_c, prot_c, p_catt)
        
        drug_co = drug_c * 0.5 + drug_sa * 0.5
        prot_co = prot_c * 0.5 + prot_sa * 0.5
        # drug_co = drug_sa 
        # prot_co = prot_sa 

        drug_att = drug_co.permute(1, 2, 0)
        prot_att = prot_co.permute(1, 2, 0)

        drugConv = self.Drug_max_pool(drug_att).squeeze(2)
        protConv = self.Prot_max_pool(prot_att).squeeze(2)
        # drug_att = drug_sa.permute(1, 2, 0)
        # prot_att = prot_sa.permute(1, 2, 0)

        # drugConv = drugConv * 0.5 + drug_att * 0.5
        # protConv = protConv * 0.5 + prot_att * 0.5

        # drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        # protConv = self.Prot_max_pool(protConv).squeeze(2)
        #end

        pair = torch.cat([drugConv, protConv], dim=1)

        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        
        return predict

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size+1,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=0,keepdim=True)
        avg_result=torch.mean(x,dim=0,keepdim=True)
        result=torch.cat([max_result,avg_result],0)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBlock(nn.Module):

    def __init__(self, channel,reduction, kernel_size):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # b, c, _ = x.size()
        
        #map add
        channel_map = self.ca(x)
        spatial_map = self.sa(x)
        att = channel_map * spatial_map
        
        #all fea
        # residual=x
        # channel_map_out = x*channel_map
        # sp = self.sa(channel_map_out)
        # out2 = sp*channel_map_out
        # out_finfa = out2+residual

        return  att

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        
        if torch.cuda.is_available():
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        else:
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
        
        #add

        self.att = nn.Linear(hid_dim, hid_dim)

        self.att_case = nn.Linear(hid_dim, 160)

    def forward(self, query, key, value, atten, mask=None):
        if len(query.shape)>len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]
        #add

        if atten.shape[1] == 20:
            atten = self.att_case(atten)
            atten_out = atten.view(bsz, -1, self.n_heads, 160 // self.n_heads).permute(0, 2, 1, 3)
            # print(atten.shape)
        else:
            atten = self.att(atten)
            atten_out = atten.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        # print(atten_out.shape)

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2) ) / self.scale 
        
        energy = energy + atten_out 

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1)
        
        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x
