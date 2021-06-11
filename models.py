"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import random
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PointNetConfig:
    """ base PointNet config """
    def __init__(self, embeddingSize, numberofPoints, numberofVars, 
                    numberofYs, method='GPT', varibleEmbedding='NOT_VAR', 
                    **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints # number of points
        self.numberofVars = numberofVars # input dimension (Xs)
        self.numberofYs = numberofYs # output dimension (Ys)
        self.method = method
        self.varibleEmbedding = varibleEmbedding

        for k,v in kwargs.items():
            setattr(self, k, v)

class PointNet(nn.Module):
    """
    :param x: a tensor of (x,y)s, it should be [batch_size, maximum_number_of_points, maximum_num_of_variables+numberofoutputs]

    we would return yet another embedding for the GPT2, it should be [batch_size, embedding_size]
    :return:

    Model: 
        Given input {(x1,y1), (x2, y2), ... (xn, yn)}

        Set up a (feed forward) network h that takes in (xi, yi) and outputs a vector h_i

        Feed all data points into that so you get 
        {h1, h2, ... hn} = (h(x1, y1), h(x2, y2), ... h(xn, yn))

        Then pass this into an order invariant pooling function, like max or sum or avg, so you get (for example using max)
        u = max(h1, h2, ..., hn)

        Then set up another network g that takes in j and outputs the output embedding y
        y = g(u)

        So in other words, we learn two networks: h and g
        and
        y = g( max((h(x1, y1), ...., h(xn, yn)) )
    """
    def __init__(self, config):
        super().__init__()

        self.unSqDim = 1
        #self.hList = nn.ModuleList()
        #for i in range(config.numberofPoints):
        #    hi = nn.Linear(config.numberofVars+config.numberofYs, config.embeddingSize, bias=False)
        #    self.hList.append(hi)
        self.hDense = nn.Linear(config.numberofVars+config.numberofYs, config.embeddingSize, bias=False)

        self.g = nn.Linear(config.embeddingSize, config.embeddingSize, bias=False)

        self.iNorm = nn.LayerNorm(config.numberofVars+config.numberofYs)
        self.lNorm = nn.LayerNorm(config.embeddingSize)

    def forward(self, points, targets=None):
        hList = []
        for pointIdx in range(points.shape[-1]):
            
            # normalize features, now the normalization is based on the x-max/max-min
            point = points[:,:,pointIdx] #self.iNorm(points[:,:,pointIdx]) #TODO: make sure this is correct
            
            h = self.hDense(point)
            #hi = hi.unsqueeze(self.unSqDim)
            #print('h shape: {}'.format(hi.shape))
            hList.append(h)

        h = torch.stack(hList, dim=self.unSqDim)
        h, h_indexes = torch.max(h, dim=self.unSqDim, keepdim=False) # order invariant pooling

        g = self.g(h)
        g = self.lNorm(g)

        return g

# pointNet based on Convolution, T-NET naming is not accurate
class tNet(nn.Module):
    """
    The PointNet structure in the orginal PointNet paper: 
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et. al. 2017
    """
    def __init__(self, config):
        super(tNet, self).__init__()

        self.activation_func = F.relu
        self.num_units = config.embeddingSize

        self.conv1 = nn.Conv1d(config.numberofVars+config.numberofYs, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        #self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars+config.numberofYs)
        #self.input_layer_norm = nn.LayerNorm(config.numberofPoints)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        """
        :param x: [batch, #features, #points]
        :return:
            logit: [batch, embedding_size]
        """
        x = self.input_batch_norm(x)
        x = self.activation_func(self.bn1(self.conv1(x)))
        x = self.activation_func(self.bn2(self.conv2(x)))
        x = self.activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units

        x = self.activation_func(self.bn4(self.fc1(x)))
        x = self.activation_func(self.bn5(self.fc2(x)))
        #x = self.fc2(x)

        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, pointNetConfig=None):
        super().__init__()

        self.config = config
        self.pointNetConfig = pointNetConfig
        self.pointNet = None

        embeddingSize = config.n_embd
        if self.pointNetConfig is not None:            

            if self.pointNetConfig.method == 'EMB_CAT':
                print('The model is going to concatenate the embeddings!')
                embeddingSize = config.n_embd//2 # if concatenation

            # OVERRIDE: POINT embedding should have the same size of token and position embedding
            if self.pointNetConfig.embeddingSize != embeddingSize:
                print("We've override your choice for pointNet embedding! Updating {} with {}!".format(self.pointNetConfig.embeddingSize, embeddingSize))
                self.pointNetConfig.embeddingSize = embeddingSize   

            self.pointNet = tNet(self.pointNetConfig)
            #self.pointNet = PointNet(self.pointNetConfig)

            self.vars_emb = nn.Embedding(self.pointNetConfig.numberofVars+1, embeddingSize)
            
        if self.pointNetConfig.method == 'EMB_CON':
            print('Add one to the supported block size!')
            self.block_size = config.block_size + 1 # add a first token
            config.block_size += 1
        else:        
            self.block_size = config.block_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, embeddingSize, padding_idx=self.config.padding_idx)        
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, embeddingSize))
        self.drop = nn.Dropout(config.embd_pdrop)        

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        if self.pointNetConfig.method == 'OUT_CAT':
            self.head = nn.Linear(config.n_embd*2, config.vocab_size, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, points=None, variables=None, tokenizer=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        if points != None and self.pointNet !=None:
            points_embeddings = self.pointNet(points)

            if variables != None and self.pointNetConfig.varibleEmbedding =='LEA_EMB':
                # add the variables information to the point embedding
                variables_embeddings = self.vars_emb(variables)
                points_embeddings += variables_embeddings

            points_embeddings = points_embeddings.unsqueeze(1)

            if self.pointNetConfig.method == 'EMB_CON':
                input_embedding = token_embeddings + position_embeddings
                input_embedding = torch.cat((points_embeddings, input_embedding), dim=1) # add point embedding as the first token
            else:
                # TODO: I have to find a smarter way to replace this tile overhead
                points_embeddings = torch.tile(points_embeddings, (1,token_embeddings.shape[1],1))

                if self.pointNetConfig.method == 'EMB_SUM':
                    # summation
                    input_embedding = token_embeddings + position_embeddings + points_embeddings
                elif self.pointNetConfig.method == 'EMB_CAT':
                    # concatenation, you have to also change the dimensionality to half
                    input_embedding = token_embeddings + position_embeddings
                    input_embedding = torch.cat((input_embedding, points_embeddings), dim=-1)
                else:
                    input_embedding = token_embeddings + position_embeddings
        else:
            input_embedding = token_embeddings + position_embeddings

        x = self.drop(input_embedding)
        x = self.blocks(x)
        x = self.ln_f(x)

        if self.pointNetConfig.method == 'OUT_SUM':
            x += points_embeddings
        elif self.pointNetConfig.method == 'OUT_CAT':
            x = torch.cat((x, points_embeddings), dim=-1)
        elif self.pointNetConfig.method == 'EMB_CON':
            # remove the first token
            x = x[:,1:,:]

        logits = self.head(x)

        printCondition = random.random() < 0.001 and tokenizer is not None
        if printCondition:
            Input, Logit = idx[0], logits.max(-1)[1][0]
            
            InputChr = ''.join([tokenizer[int(i)] for i in Input])
            LogitChr = ''.join([tokenizer[int(i)] for i in Logit])
            
            print('Input:{}\nLogit:{}'.format(Input, Logit))
            print('Input:{}\nLogit:{}'.format(InputChr, LogitChr)) 

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if printCondition:
                Target = targets[0]
                TargetChr = ''.join([tokenizer[int(i)] for i in Target])
                print('Target:{}'.format(Target))
                print('Target:{}'.format(TargetChr)) 

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                   ignore_index=self.config.padding_idx)

        return logits, loss
