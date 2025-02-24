import math
import numpy as np
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.nn import functional as F

################################################################

class ResidualNorm (nn.Module):
    def __init__ (self, size, dropout):
        super(ResidualNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MLP (nn.Module):
    def __init__(self, model_depth, ff_depth, dropout):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(model_depth, ff_depth)
        self.w2 = nn.Linear(ff_depth, model_depth)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))    

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualFF (nn.Module):
    def __init__ (self, conv_dim, mlp_dim, n_targets, activation, dropout):
        super(ResidualFF, self).__init__()

        self.conv_out = nn.Linear(conv_dim, mlp_dim)
        self.conv_out_norm = LayerNorm(mlp_dim)
        self.activation = activation
        self.ff1 = MLP(mlp_dim, mlp_dim, dropout)
        self.resnorm1 = ResidualNorm(mlp_dim, dropout)
        self.out = nn.Linear(mlp_dim, n_targets)

    def forward (self, x):
        x = self.conv_out(x)
        x = self.conv_out_norm(self.activation(x))
        x = self.resnorm1(x, self.ff1)
        y = self.out(x)
        return y

################################################################

def tanhexp (x):
    return x * F.relu(torch.exp(x))

class TanhExp (nn.Module):
    def __init__ (self):
        super(TanhExp, self).__init__()
        pass
    
    def forward (self, x):
        return tanhexp(x)
    
def swish (x):
    return x * F.sigmoid(x)

class Swish (nn.Module):
    def __init__ (self):
        super(Swish, self).__init__()
        pass
    
    def forward (self, x):
        return swish(x)
    
################################################################

class Net (nn.Module):
    def init_weights (self, layer):
        #nn.init.xavier_uniform_(layer.weight)
        #layer.bias.data.fill_(0.01)
        return layer

    def __init__ (self, n_targets, n_descr, filters, mlp_dim, dropout, activation, vocsize=None, embdim=None):
        super(Net, self).__init__()
        if vocsize is None and embdim is None:
            raise Exception('Define `vocsize` or `embdim`')

        if vocsize is None: # working with external encoder
            self.embedder = None
        else: # working with strings, build own embedder            
            self.embedder = Embedding(vocsize, embdim)

        self.n_descr = n_descr
            
        activation_module = { 'relu' : nn.ReLU(),
                              'tanhexp' : TanhExp(),
                              'swish' : Swish()}[activation]

        activation_fn = { 'relu' : F.relu,
                          'tanhexp' : tanhexp,
                          'swish' : swish}[activation]
                
        convs = []
        for nf,fs in filters:
            convs.append(
                nn.Sequential(
                    self.init_weights(
                        nn.Conv1d(embdim, nf, fs, bias=True)),
                    activation_module))

        self.outdim = np.sum([f[0] for f in filters])+n_descr*2

        self.convs = nn.ModuleList(convs)
        self.conv_drop = nn.Dropout(dropout)
        self.conv_norm = LayerNorm(self.outdim)

        if n_descr > 0:
            self.descr_block = nn.Sequential(
                nn.Linear(n_descr, n_descr*2),
                activation_module,
                nn.Dropout(dropout),
                LayerNorm(n_descr*2))

            self.ff = ResidualFF(self.outdim, mlp_dim, n_targets, activation_fn, dropout/2)

        else:
            self.ff = nn.Linear(self.outdim, n_targets)
            


        
    def forward (self, x, d):
        if self.embedder is not None:
            x = self.embedder(x)
            
        x = x.transpose(1,2)
        xcat = []
        
        if self.n_descr > 0:
            d = self.descr_block(d)
            xcat.append(d)
        
        for conv in self.convs:
            x2 = conv(x)
            x2 = x2.max(dim=2)[0]
            xcat.append(x2)

        x = torch.cat(xcat, dim=1)
        x = self.conv_norm(x)
        x = self.conv_drop(x)
        x = self.ff(x)
        return x

class Embedding (nn.Module):
    def __init__ (self, vocab_size, model_depth):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, model_depth)
        self.model_depth = model_depth
        self.positional = PositionalEncoding(model_depth)

    def forward (self, x):
        emb = self.lut(x) * math.sqrt(self.model_depth)
        return self.positional(emb)

class PositionalEncoding (nn.Module):
    def __init__ (self, model_depth, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, model_depth)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, model_depth, 2) *
                             -(math.log(10000.0) / model_depth))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    
################################################################
    

class Highway (nn.Module):
    def __init__(self,
                 input_size,
                 dropout=0,
                 gate_bias=-2,
                 activation_function=F.relu,
                 gate_activation=torch.sigmoid):

        super(Highway, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)
        self.normal_drop = nn.Dropout(dropout)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_drop = nn.Dropout(dropout)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.normal_drop(self.activation_function(self.normal_layer(x)))
        gate_layer_result = self.gate_drop(self.gate_activation(self.gate_layer(x)))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)
