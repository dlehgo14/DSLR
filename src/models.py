import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, GRU as GRU, ReLU, Softplus, Sigmoid, BatchNorm1d
from torch.autograd import Variable, grad

def make_packer(n, n_f):
    def pack(x):
        return x.reshape(-1, n_f*n)
    return pack

def make_unpacker(n, n_f):
    def unpack(x):
        return x.reshape(-1, n, n_f)
    return unpack

class GN(MessagePassing):
    def __init__(self, n_f, n_r_f, n_fr_f, msg_dim, ndim, hidden=300, aggr='add', layer_norm = False, sparsity_mode = False, sparsity_prior = 0.9, test = False):
        super(GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.n_f = n_f
        self.n_r_f = n_r_f
        self.n_fr_f = n_fr_f

        self.sparsity_mode = sparsity_mode
        self.sparsity_prior = sparsity_prior
        self.test = test
        
        
        self.msg_fnc1 = Seq(
            Lin(2*n_f+n_fr_f, hidden),
            ReLU(),
        )
        
        self.msg_fnc2 = Seq(
            Lin(hidden + 2*n_f+n_fr_f, hidden),
            ReLU(),
        )
        
        self.msg_fnc3 = Seq(
            Lin(hidden * 2, hidden),
            ReLU(),
        )
        
        self.msg_fnc4 = Seq(
            Lin(hidden * 2, msg_dim),
        )
        
        self.node_fnc1 = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
        )
        
        self.node_fnc2 = Seq(
            Lin(hidden + msg_dim+n_f, hidden),
            ReLU(),
        )
        
        self.node_fnc3 = Seq(
            Lin(hidden * 2, hidden),
            ReLU(),
        )
        
        self.node_fnc4 = Seq(
            Lin(hidden * 2, ndim),
        )

    #[docs]
    def forward(self, x, edge_index):
        #x is [n, n_f + n_rf_s]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)

        if self.sparsity_mode:
            self.c_sort = self.c.sort(descending = True).values
            self.c = self.c.reshape([-1,1])
            self.eps_c = torch.randn_like(self.c).cuda()
        tmp = tmp.cuda()
        tmp = torch.cat([tmp, self.relation[:tmp.shape[0]]], dim=1)
        
        force_residual1 = self.msg_fnc1(tmp)
        force_residual2 = self.msg_fnc2(torch.cat([force_residual1, tmp], dim = -1))
        force_residual3 = self.msg_fnc3(torch.cat([force_residual2, force_residual1], dim = -1))
        self.force = self.msg_fnc4(torch.cat([force_residual3, force_residual2], dim = -1))
        
        if self.test:
            return self.force
        elif self.sparsity_mode:
            return self.force * (self.c * self.eps_c + 1)
        else:
            return self.force

    def update(self, aggr_out, x=None):
        tmp = torch.cat([x, aggr_out], dim=1)
        future_node_residual1 = self.node_fnc1(tmp)
        future_node_residual2 = self.node_fnc2(torch.cat([future_node_residual1, tmp], dim = -1))
        future_node_residual3 = self.node_fnc3(torch.cat([future_node_residual2, future_node_residual1], dim = -1))
        future_node = self.node_fnc4(torch.cat([future_node_residual3, future_node_residual2], dim = -1))
        return future_node #[n, nupdate]

class OGN(GN):
    def __init__(self, n_f, n_r_f, n_fr_f, msg_dim, ndim, edge_index, aggr='add', hidden=300, sparsity_mode = False, sparsity_prior = 0.9, test = False):

        super(OGN, self).__init__(n_f, n_r_f, n_fr_f, msg_dim, ndim, hidden=hidden, aggr=aggr, sparsity_mode = sparsity_mode, sparsity_prior = sparsity_prior, test = test)
        self.edge_index = edge_index
        self.ndim = ndim
        self.relation = None
        self.c = None

    def just_derivative(self, g, x, augment=False, augmentation=3):
        #x is [n, n_f]f
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)

        edge_index = g.edge_index
        ret = self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)
        return ret

    def loss(self, g, augment=True, square=False, augmentation=3, **kwargs):
        print('loss?')
        if square:
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        else:
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))

# Relation
class RGN(MessagePassing):
    def __init__(self, n_f, n_r_f, n_fr_f, msg_dim, ndim, sparsity_mode, hidden=300, aggr='add', test = False):
        super(RGN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.n_f = n_f
        self.n_r_f = n_r_f
        self.relation_fnc_n_layers = 4
        self.n_fr_f = n_fr_f
        self.sparsity_mode = sparsity_mode
        self.KL_eps = 1
        if test:
            self.KL_eps = 0

        if self.sparsity_mode:
            n_fr_f += 1
        
        #self.node_embedding_fnc = Seq(
        #    Lin(n_f, hidden),
        #    ReLU(),
        #    Lin(hidden, hidden),
        #    ReLU(),
        #    Lin(hidden, hidden),
        #    ReLU(),
        #    Lin(hidden, n_r_f),
        #)
        
        self.relation_fnc = GRU(input_size = 2*n_f, hidden_size = n_r_f, num_layers = self.relation_fnc_n_layers, batch_first=True, dropout = 0.0)

        self.relation_mean_fnc1 = Seq(
            Lin(n_r_f, hidden),
            ReLU(),
        )
        self.relation_mean_fnc2 = Seq(
            Lin(hidden + n_r_f, hidden),
            ReLU(),
        )
        self.relation_mean_fnc3 = Seq(
            Lin(hidden*2, hidden),
            ReLU(),
        )
        self.relation_mean_fnc4 = Seq(
            Lin(hidden*2, n_fr_f),
        )
        
        self.relation_logvar_fnc1 = Seq(
            Lin(n_r_f, hidden),
            ReLU(),
        )
        self.relation_logvar_fnc2 = Seq(
            Lin(hidden + n_r_f, hidden),
            ReLU(),
        )
        self.relation_logvar_fnc3 = Seq(
            Lin(hidden*2, hidden),
            ReLU(),
        )
        self.relation_logvar_fnc4 = Seq(
            Lin(hidden*2, n_fr_f),
        )

    #[docs]
    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f * seen]
        x_i = x_i.reshape(x_i.shape[0], -1, self.n_f)
        x_j = x_j.reshape(x_i.shape[0], -1, self.n_f)
        #batch = x_i.shape[0]
        #x_i = x_i.reshape(-1, self.n_f)
        #x_j = x_j.reshape(-1, self.n_f)
        #x_i_embedded = self.node_embedding_fnc(x_i)
        #x_j_embedded = self.node_embedding_fnc(x_j)
        
        #x_i_embedded = x_i_embedded.reshape(batch, -1, self.n_r_f)
        #x_j_embedded = x_j_embedded.reshape(batch, -1, self.n_r_f)
        
        tmp = torch.cat([x_i, x_j], dim= -1)
        self.before_messages = torch.zeros_like(torch.empty([self.relation_fnc_n_layers, tmp.shape[0],self.n_r_f])).cuda()
        tmp = tmp.cuda()
        #ret = self.msg_fnc(tmp)
        #ret = ret.view(ret.shape[0], 1, ret.shape[1])
        self.before_messages = self.before_messages.cuda()
        self.edge_state, self.before_messages = self.relation_fnc(tmp, self.before_messages)
        self.edge_state = self.edge_state[:,-1,:]

        mean_residual1 = self.relation_mean_fnc1(self.edge_state)
        mean_residual2 = self.relation_mean_fnc2(torch.cat([mean_residual1, self.edge_state], dim = -1))
        mean_residual3 = self.relation_mean_fnc3(torch.cat([mean_residual2, mean_residual1], dim = -1))
        self.mean = self.relation_mean_fnc4(torch.cat([mean_residual3, mean_residual2], dim = -1))
        
        logvar_residual1 = self.relation_logvar_fnc1(self.edge_state)
        logvar_residual2 = self.relation_logvar_fnc2(torch.cat([logvar_residual1, self.edge_state], dim = -1))
        logvar_residual3 = self.relation_logvar_fnc3(torch.cat([logvar_residual2, logvar_residual1], dim = -1))
        self.logvar = self.relation_logvar_fnc4(torch.cat([logvar_residual3, logvar_residual2], dim = -1))
        
        self.std = torch.exp(0.5*self.logvar)
        self.eps = torch.randn_like(self.std).cuda() * self.KL_eps
        self.c = self.mean[:, -1]

        self.relation = self.mean + self.std * self.eps

        if self.sparsity_mode:
            self.relation = self.relation[:, :-1]
            self.mean = self.mean[:,:-1]
            self.std = self.std[:,:-1]
            self.logvar = self.logvar[:, :-1]
            self.c = torch.sigmoid(self.c)

        return self.relation

    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]
        return None

class ROGN(RGN):
    def __init__(self, n_f, n_r_f, n_fr_f, msg_dim, ndim, sparsity_mode, edge_index, aggr='add', test = False, hidden=300, nt=1):

        super(ROGN, self).__init__(n_f, n_r_f, n_fr_f, msg_dim, ndim, sparsity_mode, hidden=hidden, aggr=aggr, test = test)
        self.edge_index = edge_index
        self.ndim = ndim
        self.relation = None

    def just_derivative(self, g, x, augmentation=3, augment = False):
        #x is [n * n(edge_index), n_f * seen]
        ndim = self.ndim
        edge_index = g.edge_index
        if augment:
            n_see = x.shape[-1]//self.n_f
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), n_see).to(x.device)

            d = torch.arange(ndim) #[0,1]
            d = d.repeat(n_see) #[0,1,0,1, ...]
            e = torch.arange(n_see) #[0,1,...,n_see-1]
            e = e.view([n_see,1]) # [ [0] , [1], ..., [n_see-1] ]
            e = e.repeat(1,ndim).flatten() * self.n_f # [0,0, n_f, n_f, ..., (n_see-1)*n_f, (n_see-1)*n_f ]
            f = d+e # [0,1, n_f, n_f+1, ..., (n_see-1)*n_f, (n_see-1)*n_f+1 ]
            x = x.index_add(1, f.to(x.device), augmentation)
        ret = self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)
        return ret

    def loss(self, g, augment=True, square=False, augmentation=3, **kwargs):
        if square:
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        else:
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))
