from src.simulate import *
from src.models import *
from src.datasets_list import datasets
import numpy as np
import torch
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, StepLR
from sklearn.decomposition import IncrementalPCA
import os
import time

cmu_edges = [[ 0,  1],
 [ 0,  6],
 [ 0, 11],
 [ 1,  2],
 [ 2,  3],
 [ 3,  4],
 [ 4,  5],
 [ 6,  7],
 [ 7,  8],
 [ 8,  9],
 [ 9, 10],
 [11, 12],
 [12, 13],
 [13, 14],
 [13, 17],
 [13, 24],
 [14, 15],
 [15, 16],
 [17, 18],
 [18, 19],
 [19, 20],
 [20, 21],
 [20, 23],
 [21, 22],
 [24, 25],
 [25, 26],
 [26, 27],
 [27, 28],
 [27, 30],
 [28, 29]]

def loss_visualization(train_name, data_name):
    path = './models/%s/%s'%(train_name, data_name)
    train_f = open(path + '/train_loss.log','r')
    test_f = open(path + '/test_loss.log','r')

    train_epochs = []
    train_loss = []
    test_epochs = []
    test_loss = []
    while True:
      train_line = train_f.readline()
      if train_line == '': break
      train_epochs.append(int(train_line.split(' ')[0])+1)
      train_loss.append(float(train_line.split(' ')[1]))
    while True:
      test_line = test_f.readline()
      if test_line == '': break
      test_epochs.append(int(test_line.split(' ')[0])+1)
      test_loss.append(float(test_line.split(' ')[1]) * 10)
    print('epochs: %d'%(len(train_epochs)))
    print('best epoch: %d'%np.argmin(test_loss))
    train_f.close()
    test_f.close()

    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = './results/%s'%(train_name)
    if not os.path.exists(path):
        os.mkdir(path)
    path = './results/%s/%s'%(train_name, data_name)
    if not os.path.exists(path):
        os.mkdir(path)

    plt.plot(train_epochs, train_loss, c='red', label='train')
    plt.plot(test_epochs, test_loss, c='blue', label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path + '/loss.png')
    return np.min(test_loss)

def get_edge_index(n, sim):
    adj = (np.ones((n, n)) - np.eye(n)).astype(int)
    edge_index = torch.from_numpy(np.array(np.where(adj)))
    return edge_index

def train_data_load(args, data_params):
    nr = data_params['nr']
    n = data_params['n']
    t_seen = data_params['t_seen']
    ns = data_params['ns']
    dim = data_params['dim']
    nt = data_params['nt']
    dt = data_params['dt']
    sim = data_params['sim']
    t_interval = data_params['t_interval']
    batch = args.batch_size

    data_name = "r%d_o%d_tseen%d_ns%d_dim%d_nt%d_dt%g_%s"%(nr,n,t_seen,ns,dim,nt,dt,sim)
    load = np.load('./data/%s.npz'%data_name)
    data = load['data']
    data_valid = load['data_valid']
    #data_valid = load['data']

    # Normalize data
    data = data.reshape([ns, 1, n, t_seen, -1])
    data_dim = data.shape[-1]
    data_valid = data_valid.reshape([data_valid.shape[0], data_valid.shape[1], data_valid.shape[2], -1, data_dim])
    loc_max = data[:,:,:,:,:dim].max()
    loc_min = data[:,:,:,:,:dim].min()
    vel_max = data[:,:,:,:,dim:dim*2].max()
    vel_min = data[:,:,:,:,dim:dim*2].min()

    # Calcualte the change of nodes' states that will be used as ground truth of relation decoder.
    data[:,:,:,:,:dim] = (data[:,:,:,:,:dim] - loc_min) * 2 / (loc_max - loc_min) - 1
    data[:,:,:,:,dim:dim*2] = (data[:,:,:,:,dim:dim*2] - vel_min) * 2 / (vel_max - vel_min) - 1
    data_valid[:,:,:,:,:dim] = (data_valid[:,:,:,:,:dim] - loc_min) * 2 / (loc_max - loc_min) - 1
    data_valid[:,:,:,:,dim:dim*2] = (data_valid[:,:,:,:,dim:dim*2] - vel_min) * 2 / (vel_max - vel_min) - 1

    data = data.reshape([data.shape[0], data.shape[1], data.shape[2], -1])
    data_valid = data_valid.reshape([data_valid.shape[0], data_valid.shape[1], data_valid.shape[2], -1])

    # Calcualte the change of nodes' states that will be used as ground truth of relation decoder.
    data_y = data.copy()
    data_y = data_y.reshape([ns, 1, n, t_seen, -1])
    data_y[:, :, :, :, -2] = 0 # Erase the code for relation.
    data_dim = data_y.shape[-1]
    data_y = (data_y[:, :, :, 1:, :] - data_y[:, :, :, :-1, :]) / (dt * t_interval)
    data_y = data_y.reshape([ns, 1, n, -1])

    data_y_valid = data_valid.copy()
    data_y_valid = data_y_valid.reshape([data_y_valid.shape[0], 1, n, -1, data_dim])
    data_y_valid[:, :, :, :, -2] = 0 # Erase the code for relation.
    data_y_valid = (data_y_valid[:, :, :, 1:, :] - data_y_valid[:, :, :, :-1, :]) / (dt * t_interval)
    data_y_valid = data_y_valid.reshape([data_y_valid.shape[0], 1, n, -1])

    data = data.reshape([ns, 1, n, t_seen, -1])
    if not args.test:
        data[:, :, :, :, -2] = 0 # Erase the code for relation during training.
    data = data[:, :, :, :-1, :] # Erase the last time-step which don't have valid ground truth.
    data = data.reshape([ns, 1, n, -1])

    data_valid = data_valid.reshape([data_y_valid.shape[0], 1, n, -1, data_dim])
    if not args.test:
        data_valid[:, :, :, :, -2] = 0 # Erase the code for relation during training.
    data_valid = data_valid[:, :, :, :-1, :] # Erase the last time-step which don't have valid ground truth.
    data_valid = data_valid.reshape([data_y_valid.shape[0], 1, n, -1])

    X_train = torch.from_numpy(np.concatenate(data,0))
    y_train = torch.from_numpy(np.concatenate(data_y,0))
    X_valid = torch.from_numpy(np.concatenate(data_valid,0)[:50000])
    y_valid = torch.from_numpy(np.concatenate(data_y_valid,0)[:50000])

    edge_index = get_edge_index(n, sim)

    if batch == 0:
        batch = int(64 * (4 / n)**2)

    X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], -1, data_dim])[:, :, :, :dim*2].reshape([X_train.shape[0], X_train.shape[1], -1])
    X_valid = X_valid.reshape([X_valid.shape[0], X_valid.shape[1], -1, data_dim])[:, :, :, :dim*2].reshape([X_valid.shape[0], X_valid.shape[1], -1])
    y_train = y_train.reshape([y_train.shape[0], y_train.shape[1], -1, data_dim])[:, :, :, :dim*2].reshape([y_train.shape[0], y_train.shape[1], -1])
    y_valid = y_valid.reshape([y_valid.shape[0], y_valid.shape[1], -1, data_dim])[:, :, :, :dim*2].reshape([y_valid.shape[0], y_valid.shape[1], -1])

    print('shape of X_train: ', X_train.shape)
    print('shape of y_train: ', y_train.shape)
    print('shape of X_valid: ', X_valid.shape)
    print('shape of y_valid: ', y_valid.shape)

    trainloader = DataLoader(
        [Data(
            Variable(X_train[i]),
            edge_index=edge_index,
            y=Variable(y_train[i])) for i in range(len(y_train))],
        batch_size=batch,
        shuffle=True
    )
    validloader = DataLoader(
        [Data(
            X_valid[i],
            edge_index=edge_index,
            y=y_valid[i]) for i in range(len(y_valid))],
        batch_size=1,
        shuffle=True
    )
    return X_train, y_train, X_valid, y_valid, trainloader, validloader, data, data_y, data_valid, data_y_valid, edge_index, batch, data_name

def test_data_load(args, data_params):
    nr = data_params['nr']
    n = data_params['n']
    t_seen = data_params['t_seen']
    ns = data_params['ns']
    dim = data_params['dim']
    nt = data_params['nt']
    dt = data_params['dt']
    sim = data_params['sim']
    t_interval = data_params['t_interval']
    batch = args.batch_size

    data_name = "r%d_o%d_tseen%d_ns%d_dim%d_nt%d_dt%g_%s"%(nr,n,t_seen,ns,dim,nt,dt,sim)
    load = np.load('./data/%s.npz'%data_name)
    data = load['data']
    data_test = load['data_test']

    # Normalize data
    data = data.reshape([ns, 1, n, t_seen, -1])
    data_dim = data.shape[-1]
    data_test = data_test.reshape([data_test.shape[0], data_test.shape[1], data_test.shape[2], -1, data_dim])
    loc_max = data[:,:,:,:,:dim].max()
    loc_min = data[:,:,:,:,:dim].min()
    vel_max = data[:,:,:,:,dim:dim*2].max()
    vel_min = data[:,:,:,:,dim:dim*2].min()

    # Calcualte the change of nodes' states that will be used as ground truth of relation decoder.
    data_test[:,:,:,:,:dim] = (data_test[:,:,:,:,:dim] - loc_min) * 2 / (loc_max - loc_min) - 1
    data_test[:,:,:,:,dim:dim*2] = (data_test[:,:,:,:,dim:dim*2] - vel_min) * 2 / (vel_max - vel_min) - 1

    data_test = data_test.reshape([data_test.shape[0], data_test.shape[1], data_test.shape[2], -1])

    data_y_test = data_test.copy()
    data_y_test = data_y_test.reshape([data_y_test.shape[0], 1, n, -1, data_dim])
    data_y_test[:, :, :, :, -2] = 0
    data_y_test = (data_y_test[:, :, :, 1:, :] - data_y_test[:, :, :, :-1, :]) / (dt * t_interval)
    data_y_test = data_y_test.reshape([data_y_test.shape[0], 1, n, -1])

    data_test = data_test.reshape([data_y_test.shape[0], 1, n, -1, data_dim])
    data_test = data_test[:, :, :, :-1, :]
    data_test = data_test.reshape([data_y_test.shape[0], 1, n, -1])

    X_test = torch.from_numpy(np.concatenate(data_test,0)[:50000])
    y_test = torch.from_numpy(np.concatenate(data_y_test,0)[:50000])
    edge_index = get_edge_index(n, sim)

    print('shape of X_test: ', X_test.shape)
    print('shape of y_test: ', y_test.shape)

    testloader = DataLoader(
        [Data(
            X_test[i],
            edge_index=edge_index,
            y=y_test[i]) for i in range(len(y_test))],
        batch_size=batch,
        shuffle=True
    )
    return testloader, data_test, data_y_test, edge_index, data_name

def test_model_load(n_f, n_r_f, n_fr_f, msg_dim, hidden, aggr, train_name, data_name, args, data_params):
    n = data_params['n']
    dim = data_params['dim']*2
    sim = data_params['sim']
    sparsity_mode = args.connection_value
    sparsity_prior = args.sparsity_prior

    path = './models/%s/%s'%(train_name, data_name)
    ogn = OGN(n_f, n_r_f, n_fr_f, msg_dim, dim, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr, sparsity_mode = sparsity_mode, sparsity_prior = sparsity_prior, test = args.test).cuda()
    rogn = ROGN(n_f, n_r_f, n_fr_f, msg_dim, dim, sparsity_mode, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr, test = args.test).cuda()

    ogn.load_state_dict(torch.load(path + '/decoder.pth'))
    rogn.load_state_dict(torch.load(path + '/encoder.pth'))
    return ogn, rogn

def model_load(trainloader, n_f, n_r_f, n_fr_f, msg_dim, hidden, aggr, init_lr, args, data_params):
    n = data_params['n']
    dim = data_params['dim']*2
    sim = data_params['sim']
    sparsity_mode = args.connection_value
    sparsity_prior = args.sparsity_prior
    total_epochs = args.epochs

    ogn = OGN(n_f, n_r_f, n_fr_f, msg_dim, dim, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr, sparsity_mode = sparsity_mode, sparsity_prior = sparsity_prior, test = args.test).cuda()
    rogn = ROGN(n_f, n_r_f, n_fr_f, msg_dim, dim, sparsity_mode, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr).cuda()

    opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=1e-8)
    ropt = torch.optim.Adam(rogn.parameters(), lr=init_lr, weight_decay=1e-8)

    batch_per_epoch = len(trainloader)
    sched = OneCycleLR(opt, max_lr=init_lr,
                       steps_per_epoch=batch_per_epoch,
                       epochs=total_epochs, final_div_factor=1e5)
    rsched = OneCycleLR(ropt, max_lr=init_lr,
                       steps_per_epoch=batch_per_epoch,
                       epochs=total_epochs, final_div_factor=1e5)
    return ogn, rogn, opt, ropt, sched, rsched, total_epochs, batch_per_epoch

def model_save(ogn, rogn, train_name, data_name):
    path = './models'
    if not os.path.exists(path):
        os.mkdir(path)
    path = './models/%s'%(train_name)
    if not os.path.exists(path):
        os.mkdir(path)
    path = './models/%s/%s'%(train_name, data_name)
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(ogn, path + '/decoder.pth')
    torch.save(rogn, path + '/encoder.pth')

def loss_save(epoch, train_name, data_name, dic):
    path = './models'
    if not os.path.exists(path):
        os.mkdir(path)
    path = './models/%s'%(train_name)
    if not os.path.exists(path):
        os.mkdir(path)
    path = './models/%s/%s'%(train_name, data_name)
    if not os.path.exists(path):
        os.mkdir(path)
    if epoch == 0:
        for key in list(dic.keys()):
            f = open(path + '/%s.log'%(key), 'w')
            f.close()
    for key in list(dic.keys()):
        if dic[key]:
            f = open(path + '/%s.log'%(key), 'a')
            f.write('%d %.5f\n'%(epoch,dic[key]))
            f.close()

def camera_save(camera, i, train_name, data_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = './results/%s'%(train_name)
    if not os.path.exists(path):
        os.mkdir(path)
    path = './results/%s/%s'%(train_name, data_name)
    if not os.path.exists(path):
        os.mkdir(path)
    camera.animate().save(path + '/%02d.mp4'%i)

def relation_fig_save(fig, train_name, data_name, suffix = ''):
    #fig.xlim(-0.002, 0.002)
    #fig.ylim(-0.002, 0.002)
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = './results/%s'%(train_name)
    if not os.path.exists(path):
        os.mkdir(path)
    path = './results/%s/%s'%(train_name, data_name)
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(path + '/relation' + suffix + '.png')

def get_NP_loss(self, g, x, dt, t_interval, n_decoder, augment=False, augmentation=3, print_ = False, comparative_before_messages = None, t = 0, dim =2, return_pred = False, test = False):
    if test:
        #pred = self.just_derivative(g, x, augment=augment, augmentation = augmentation)[:, :dim]
        #base_loss = torch.sum(torch.abs( g.y[:,t*dim : (t+1)*dim] - pred))
        base_loss = 0
        for i in range(1):
            pred = self.just_derivative(g, x, augment=augment, augmentation = augmentation)
            base_loss += torch.sum(torch.abs( g.y[:,(t+i)*dim : (t+i+1)*dim] - pred))
            # x: [pos * dim, vel * dim, acc * dim, etc]
            x = x + pred * dt * t_interval
    else:
        base_loss = 0
        for i in range(n_decoder):
            pred = self.just_derivative(g, x, augment=augment, augmentation = augmentation)
            base_loss += torch.sum(torch.abs( g.y[:,(t+i)*dim : (t+i+1)*dim] - pred))
            # x: [pos * dim, vel * dim, acc * dim, etc]
            x = x + pred * dt * t_interval
    if return_pred:
        return base_loss, pred
    else:
        return base_loss

def simulate(ogn, rogn, y, e, n, n_f, dim, dt, t_interval, step_size, cmu = False, t = 0):
    y = y.reshape(n, n_f).astype(np.float32)
    cur = Data(
        x=torch.from_numpy(y).cuda(),
        edge_index=e
    )
    start_time = time.time()
    dv = ogn.just_derivative(cur, cur.x).cpu().detach().numpy()
    end_time = time.time()

    y = y + dv * dt * t_interval * step_size
    return y, end_time - start_time

def video_generate_2d(train_name, data_name, i, ogn, rogn, x, e, steps, args, data_params, step_size = 1., cmu = False, length_of_tails = 75, video_step = 1, axis0 = 0, axis1 = 1, xylim = 10, relation_axis0 = 0, relation_axis1 = 1, video_save = True, bball = False):
    t_max_see = args.t_max_see
    t_seen_interval = args.t_seen_interval
    dim = data_params['dim']
    dt = data_params['dt']
    t_interval = data_params['t_interval']

    if video_save: print('\r%d-th video generating...'%(i+1, ), end = '')
    else: print('\r%d-th data testing...'%(i+1, ), end = '')

    n = x.shape[0]
    n_f = x.shape[-1]
    if cmu:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    elif bball:
        fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    camera = Camera(fig)

    # relation reasoning
    x_relation = x[:, :t_max_see:t_seen_interval, :]
    x_relation = x_relation.reshape([x_relation.shape[0],-1])
    x_relation = torch.from_numpy(x_relation).cuda()
    x_relation = Data(
        x=x_relation,
        edge_index=e
    )
    rogn.just_derivative(x_relation, x_relation.x)
    ogn.relation = rogn.relation
    ogn.c = rogn.c

    times = np.arange(int(steps//step_size))
    times = times/len(times)

    x_output = x[:, t_max_see:, :]
    x_output = x_output.transpose(1, 0, 2)
    x_input = torch.from_numpy(x_output[0,:,:]).cuda()
    x_pred = [x_input.cpu().detach().numpy()]
    total_time = 0
    for t in range(len(times)):
        cur_x, cur_time = simulate(ogn, rogn, x_pred[-1], e, n, n_f, dim, dt, t_interval, step_size, cmu, t = t)
        x_pred.append(cur_x)
        total_time += cur_time

        # edge edge_centrality_test_mode
        if (args.edge_centrality_test_mode_random or args.edge_centrality_test_mode_centrality) and t == 0:
            assert not (args.edge_centrality_test_mode_random and args.edge_centrality_test_mode_centrality),\
            'You must set only one of edge_centrality_test_mode_random or edge_centrality_test_mode_centrality True.'
            assert (args.edge_centrality_test_mode_ratio > 0 and args.edge_centrality_test_mode_ratio < 1),\
            'edge_centrality_test_mode_ratio must set in the range of [0, 1].'
            # e: [2, n(n-1)]
            # c: [n(n-1), 1]
            c = ogn.c
            edge_n = len(c)
            disconnect_n = int( edge_n * args.edge_centrality_test_mode_ratio )

            if args.edge_centrality_test_mode_centrality:
                alive_idx = (c.flatten().sort()[1])[:-disconnect_n]
            elif args.edge_centrality_test_mode_random:
                alive_idx = torch.randperm(edge_n)[:-disconnect_n]
            else:
                raise NotImplementedError('Choose the proper centerality test mode.')

            ogn.before_messages = ogn.before_messages[alive_idx]
            e = (e.transpose(1,0)[alive_idx]).transpose(1,0)
    if args.more_timesteps_simulation > 0:
        x = x_pred[-1]
        for t in range(args.more_timesteps_simulation):
            x, cur_time = simulate(ogn, rogn, x, e, n, n_f, dim, dt, t_interval, step_size, cmu, gt_acc = gt_acc, t = t)
            total_time += cur_time

    x_pred = np.array(x_pred)

    relation = ogn.relation.cpu().detach().numpy()
    connection_value = ogn.c.cpu().detach().numpy()
    relation_edges = e.transpose(1,0)

    pos_loss = 0

    start_t = 0
    if video_save and bball:
        start_t = len(times) - 1
        length_of_tails = len(times)
    for t_idx in range(start_t, len(times), video_step):
        start = max([0, t_idx - length_of_tails])
        c_times = times[start:t_idx+1]
        c_gt = x_output[ int(start*step_size) : int(t_idx*step_size)+1 ]
        c_pred = x_pred[start:t_idx+1]
        if cmu:
            pos_loss += np.sum( (c_gt[-1, :, :3] - c_pred[-1, :, :3])**2 )
        else:
            pos_loss += np.sum( (c_gt[-1, :, :2] - c_pred[-1, :, :2])**2 )
        if video_save and bball:
            c_gt_input = x[:, :t_max_see, :].transpose(1, 0, 2)
            print(c_gt_input.shape)
            print(c_gt.shape)
            c_gt = np.concatenate([c_gt_input, c_gt], axis = 0)
            c_pred = np.concatenate([c_gt_input, c_pred], axis = 0)
        if video_save:
            c = np.asarray([ [1., 0., 0.], [1., 0., 1.], [0., 1., 0.], [0., 1., 1.], [0., 0., 1.]]) # [5,3]
            c = c.reshape([1, 5, 3]).repeat(len(c_gt), 0) # [len(c_gt), 5, 3]
            alphas = np.ones_like(c[:, 0, 0])
            alphas[:t_max_see] *= 0.1
            
            for j in range(n):
                if cmu:
                    ax[0].scatter(c_gt[:, j, axis0],
                                  c_gt[:, j, axis1], color='black', s = 1)

                    ax[1].scatter(c_pred[:, j, axis0],
                                  c_pred[:, j, axis1], color='black', s = 1)
                    ax[1].scatter(c_gt[:, j, axis0],
                                  c_gt[:, j, axis1], color='red', s = 1, zorder=-1)

                    ax[2].scatter(c_pred[:, j, relation_axis0],
                              c_pred[:, j, relation_axis1], color='black', s = 5)
                    ax[3].scatter(c_pred[:, j, relation_axis0],
                            c_pred[:, j, relation_axis1], color='black', s = 5)
                elif bball:
                    ax[0].scatter(c_gt[:, j, axis0],
                                  c_gt[:, j, axis1], color=c[:, j], alpha = alphas, s = 60)

                    ax[1].scatter(c_pred[:, j, axis0],
                                  c_pred[:, j, axis1], color=c[:, j], alpha = alphas, s = 60)

                    ax[2].scatter(c_pred[:, j, axis0],
                              c_pred[:, j, axis1], color=c[:, j], alpha = alphas, s = 60)
                    ax[3].scatter(c_pred[:, j, axis0],
                            c_pred[:, j, axis1], color=c[:, j], alpha = alphas, s = 60)
                else:
                    rgba = make_transparent_color(len(c_times), j/n)
                    ax[0].scatter(c_gt[:, j, axis0],
                                  c_gt[:, j, axis1], color=rgba[::int(1/step_size)], s = 2)
                    ax[1].scatter(c_pred[:, j, axis0],
                                  c_pred[:, j, axis1], color=rgba, s = 2)
                    black_rgba = rgba
                    black_rgba[:, :3] = 0.75
            if cmu:
                for edge in cmu_edges:
                    point1 = [c_gt[-1, edge[0], axis0], c_gt[-1, edge[0],axis1]]
                    point2 = [c_gt[-1, edge[1], axis0], c_gt[-1, edge[1],axis1]]
                    ax[0].plot( [point1[0], point2[0]], [point1[1], point2[1]], c= 'grey', alpha = 1, zorder=-1, linewidth=0.3  )

                    point1 = [c_pred[-1, edge[0], axis0], c_pred[-1, edge[0],axis1]]
                    point2 = [c_pred[-1, edge[1], axis0], c_pred[-1, edge[1],axis1]]
                    ax[1].plot( [point1[0], point2[0]], [point1[1], point2[1]], c= 'grey', alpha = 1, zorder=-1, linewidth=0.3  )

                    point1 = [c_pred[-1, edge[0], relation_axis0], c_pred[-1, edge[0],relation_axis1]]
                    point2 = [c_pred[-1, edge[1], relation_axis0], c_pred[-1, edge[1],relation_axis1]]
                    ax[2].plot( [point1[0], point2[0]], [point1[1], point2[1]], c= 'grey', alpha = 1, zorder=-1 )

                    ax[3].plot( [point1[0], point2[0]], [point1[1], point2[1]], c= 'grey', alpha = 1, zorder=-1 )

                for idx, edge in enumerate(relation_edges):
                    point1 = [c_pred[-1, edge[0], relation_axis0], c_pred[-1, edge[0],relation_axis1]]
                    point2 = [c_pred[-1, edge[1], relation_axis0], c_pred[-1, edge[1],relation_axis1]]
                    #if (1-float(connection_value[idx])) <= 1.0 and (1-float(connection_value[idx])) > 0.8:
                    ax[2].plot( [point1[0], point2[0]], [point1[1], point2[1]], c=[(1-float(connection_value[idx])), 0, float(connection_value[idx])], alpha = 1., linewidth=0.02+(1-float(connection_value[idx])) * 0.5)
            elif bball:
                for idx, edge in enumerate(relation_edges):
                    point1 = [c_pred[-1, edge[0], axis0], c_pred[-1, edge[0],axis1]]
                    point2 = [c_pred[-1, edge[1], axis0], c_pred[-1, edge[1],axis1]]
                    #if (1-float(connection_value[idx])) <= 1.0 and (1-float(connection_value[idx])) > 0.8:
                    ax[2].plot( [point1[0], point2[0]], [point1[1], point2[1]], c=[(1 - float(connection_value[idx])), 0, float(connection_value[idx])], alpha = 1., linewidth=0.02+(1-float(connection_value[idx])) * 0.5)

            plt.tight_layout()
            camera.snap()
    if video_save:
        for k in range(2):
            ax[k].set_xlim(-xylim, xylim)
            ax[k].set_ylim(-xylim, xylim)
        if cmu:
            ax[2].set_xlim(-0.6, 0.2)
            ax[2].set_ylim(-0.4, 0.4)
            ax[3].set_xlim(-0.6, 0.2)
            ax[3].set_ylim(-0.4, 0.4)
        camera_save(camera, i, train_name, data_name)
    return pos_loss, total_time

def get_dataset_parameters(dataset_name):
    data_params = {}
    dataset = datasets[dataset_name]
    data_params['ns'] = dataset['ns']   # Number of training data
    data_params['n'] = dataset['n']     # Number of nodes
    data_params['dim'] = dataset['dim'] # Dimension of node state
    data_params['nt'] = dataset['nt']   # Total number of time-steps of simualtion
    data_params['dt'] = dataset['dt']   # Delta-t of simulation
    data_params['sim'] = dataset['sim'] # Name of simulation
    data_params['nr'] = dataset['nr']   # Number of relation
    data_params['t_interval'] = dataset['t_interval']   # Interval of time-steps when transforming simualted data into ready data for model input
    data_params['t_seen'] = dataset['t_seen'] # Total number of time-steps of data for model input
    data_params['s']  = SimulationDataset(dataset['sim'], n=dataset['n'], dim=dataset['dim'], nt=dataset['nt'] , dt=dataset['dt'])
    return data_params

def dir_naming(args):
    base_path = args.train_name
    if args.connection_value:
        print('sparsity mode')
        base_path += '_connectionValue'
        if args.sparsity_prior != 0.0:
            print('Use sparsity prior: %f'%args.sparsity_prior)
            base_path += '_sparsityPrior%.2f'%args.sparsity_prior
    if args.n_relation_STD != 1:
        print('#relation for relation standard deviation loss: %d'%args.n_relation_STD)
        base_path += '_nRelationSTD%d'%args.n_relation_STD
    if args.RPT:
        print('Use reparameterization trick.')
        base_path += '_RPT'
    if args.RST:
        print('Use random sampling trick.')
        base_path += '_RST'
    if args.augment:
        print('Use data augmentation')
        base_path += '_augment'
    base_path += '_seed%d'%args.seed
    return base_path
