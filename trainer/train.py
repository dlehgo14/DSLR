from src.simulate import *
from src.utils import *
from src.datasets_list import datasets
import numpy as np
import torch
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
import argparse
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

# training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train-name', type=str, default='train0',
                    help='Training id')
parser.add_argument('--dataset', type=str, default='spring_None',
                    help='Select dataset to train with.')
parser.add_argument('--t-seen-interval', type=int, default=1, help='The interval between timesteps for inferring relation staes. Lower the lower memory, higher the higher accuracy.')
parser.add_argument('--t-max-see', type=int, default=49, help='How many time-steps to watch for inferring relations states')
parser.add_argument('--sparsity-prior', type=float, default=0.0,
                    help='Sparsity prior given to DSLR. It is valid only if connection value is used. If sparsity prior set to 0.0, sparsity prior will be not used.')
parser.add_argument('--n-relation-STD', type=int, default=5, help='The number of relation states used for relation standard deviation loss. If set to 1, relation standard deviation loss will not be used.')
parser.add_argument('--n-decoder', type=int, default=10, help='The number of time-steps of node states predicted by relation decoder in training phase.')
parser.add_argument('--batch-size', type=int, default=96, help='Batch size.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use.')
parser.add_argument('--epochs', type=int, default=1000, help='Total training epochs.')
parser.add_argument('--test_per_epochs', type=int, default=1, help='How frequently test during training.')
parser.add_argument('--msg-dim', type=int, default=100, help='Dimension of message vector of relation decoder.')

parser.add_argument('--load', action='store_true', default=False,
                    help='Load or train from scratch.')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set False for training.')
parser.add_argument('--RPT', action='store_true', default=False,
                    help='If set to False, we do not use reparameterization trick in relation decoder.')
parser.add_argument('--RST', action='store_true', default=False,
                    help='If set to False, we do not use random sampling trick in relation decoder.')
parser.add_argument('--augment', action='store_true', default=False,
                    help='If or not to augment data during training.')
parser.add_argument('--connection-value', action='store_true', default=False,
                    help='If or not to use connection value.')
parser.add_argument('--use-valid', action='store_true', default=False,
                    help='Use validation set, and save the model with lowest valid loss.')
parser.add_argument('--CAT', action='store_true', default=False,
                    help='Compression avoidance trick used in fNRI.')
args = parser.parse_args()

train_name = dir_naming(args)
print(train_name)

# dataset parameters
data_params = get_dataset_parameters(args.dataset)

# model parameters
aggr = 'add'
hidden = 196#256#128 # 300
test = '_l1_'
n_f = data_params['dim'] * 2
dim = n_f
n_r_f = 128 # 300 # relation latent state의 dimension
n_fr_f = 10

# training parameters
torch.cuda.set_device(args.gpu)
init_lr = 1e-3
eps = 1e-5
sparsity_prior = 0.5

t_seen = data_params['t_seen']
dt = data_params['dt']
t_interval = data_params['t_interval']
t_max_see = args.t_max_see
t_seen_interval = args.t_seen_interval
augment = args.augment
msg_dim = args.msg_dim

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X_train, y_train, X_valid, y_valid, trainloader, validloader, data, data_y, data_valid, data_y_valid, edge_index, batch, data_name = train_data_load(args, data_params)

    #if data_compact_mode:
    #    n_f = 4
    #if state_output_mode:
    #    dim = n_f

    ogn, rogn, opt, ropt, sched, rsched, total_epochs, batch_per_epoch = model_load(trainloader, n_f, n_r_f, n_fr_f, msg_dim, hidden, aggr, init_lr, args, data_params)
    if args.load:
        ogn, rogn = test_model_load(n_f, n_r_f, n_fr_f, msg_dim, hidden, aggr, train_name, data_name, args, data_params)
        min_loss = loss_visualization(train_name, data_name)
    else:
        min_loss = None



    print('batch_per_epoch: ', batch_per_epoch)
    print('batch_size: ', batch)

    # train
    for epoch in tqdm(range(0, args.epochs)):
        ogn.cuda()
        rogn.cuda()

        i = 0
        train_NP_loss = 0.0
        train_KL_loss = 0.0
        train_SD_loss = 0.0
        train_c_loss = 0.0
        num_items = 0
        for ginput in trainloader:
            if i >= batch_per_epoch:
                break
            opt.zero_grad()
            ropt.zero_grad()

            relations = []
            NP_loss = 0
            KL_loss = 0
            c_loss = 0
            for t in range(args.n_relation_STD):
                ginput.x = ginput.x.cuda()
                ginput.y = ginput.y.cuda()
                ginput.edge_index = ginput.edge_index.cuda()
                ginput.batch = ginput.batch.cuda()

                if args.RST:
                    t_seen_start = np.random.randint(0, t_seen - t_max_see + 1)
                elif args.CAT:
                    t_seen_start = 0
                else:
                    t_seen_start = 0

                x = ginput.x.reshape([ginput.x.shape[0], -1, n_f])
                x = x[:, t_seen_start: (t_seen_start + t_max_see):t_seen_interval, :]
                x = x.reshape([x.shape[0],-1])
                rogn.just_derivative(ginput, x, augment = augment)
                relation_state = rogn.relation
                relations.append(relation_state.reshape(1, relation_state.shape[0], relation_state.shape[1]))
            
            ogn.relation = relation_state#torch.mean(torch.cat(relations), axis=0)
            ogn.c = rogn.c
            if args.RST:
                t_seen_random = np.random.randint(1, t_seen + 1 - args.n_decoder)
            else:
                t_seen_random = t_max_see
            x = ginput.x[:,n_f * (t_seen_random-1):n_f * (t_seen_random)]
            NP_loss += get_NP_loss(ogn, ginput, x, dt = dt, t_interval = t_interval, n_decoder = args.n_decoder,
                            t = t_seen_random-1,
                            comparative_before_messages = None, dim = dim, augment = augment)

            KL_loss += (-0.5 * torch.sum(1 + rogn.logvar - rogn.mean.pow(2) - rogn.logvar.exp()))
            if args.connection_value:
                if args.sparsity_prior == 0:
                    c_loss += torch.sum(- torch.log(ogn.c_sort + eps))
                else:
                    c_loss += torch.sum(- torch.log(ogn.c_sort[: int(len(ogn.c_sort) * args.sparsity_prior)] + eps))
                    c_loss += torch.sum(- torch.log(1 - ogn.c_sort[int(len(ogn.c_sort) * args.sparsity_prior):] + eps) )

            NP_loss /= 1#args.n_relation_STD
            KL_loss /= 1#args.n_relation_STD
            c_loss /= 1#args.n_relation_STD
            if args.n_relation_STD >= 2:
                SD_loss = torch.sum(torch.std(torch.cat(relations), axis=0))
            else:
                SD_loss = 0
            batch_loss = NP_loss + 0.1 * KL_loss + SD_loss + c_loss * 1e-3
            (batch_loss/int(ginput.batch[-1]+1)).backward()

            opt.step()
            ropt.step()
            sched.step()
            rsched.step()

            i += 1
            num_items += int(ginput.batch[-1]+1)
            train_NP_loss += NP_loss.item()
            train_KL_loss += KL_loss.item()
            if args.n_relation_STD >= 2:
                train_SD_loss += SD_loss.item()
            if args.connection_value:
                train_c_loss += c_loss.item()
        train_NP_loss /= num_items
        train_KL_loss /= num_items
        train_SD_loss /= num_items
        train_c_loss /= num_items
        print('L_NP: %f / L_KL: %f / L_SD: %f / L_c: %f'%(train_NP_loss, train_KL_loss, train_SD_loss, train_c_loss))

        if (epoch + 1) % args.test_per_epochs == 0:
            i = 0
            test_NP_loss = 0.0
            num_items = 0
            for ginput in validloader:
                if i >= len(validloader):
                    break
                opt.zero_grad()
                ropt.zero_grad()

                ginput.x = ginput.x.cuda()
                ginput.y = ginput.y.cuda()
                ginput.edge_index = ginput.edge_index.cuda()
                ginput.batch = ginput.batch.cuda()

                x = ginput.x.reshape([ginput.x.shape[0], -1, n_f])
                x = x[:, : t_max_see: t_seen_interval, :]
                x = x.reshape([x.shape[0],-1])
                rogn.just_derivative(ginput, x)

                ogn.relation = rogn.relation
                ogn.c = rogn.c
                x = ginput.x[:, -2 * n_f : -n_f]

                NP_loss = get_NP_loss(ogn, ginput, x, n_decoder = args.n_decoder, dim=dim, test = True, dt = dt,
                                    t_interval = t_interval, t = -2, augment = False)

                test_NP_loss += NP_loss.detach()
                i += 1
                num_items += int(ginput.batch[-1]+1)

            test_NP_loss = test_NP_loss/num_items

            if args.use_valid:
                if not min_loss or test_NP_loss < min_loss:
                    print('Lowest valid loss. Save model.')
                    min_loss = test_NP_loss
                    ogn.cpu()
                    rogn.cpu()
                    model_save(ogn.state_dict(), rogn.state_dict(), train_name, data_name)
            else:
                model_save(ogn.state_dict(), rogn.state_dict(), train_name, data_name)
        else:
            test_NP_loss = None

        loss_save(epoch, train_name, data_name, {'train_loss':train_NP_loss, 'test_loss':test_NP_loss})

if __name__ == '__main__':
    main()
