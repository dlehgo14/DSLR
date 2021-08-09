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
from sklearn.decomposition import PCA

# test parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train-name', type=str, default='train0',
                    help='Training id')
parser.add_argument('--dataset', type=str, default='spring_None',
                    help='Select dataset to train with.')
parser.add_argument('--clustering-model', type=str, default='kmeans',
                    help='Choose clustering model (kmeans or GMM).')
parser.add_argument('--t-seen-interval', type=int, default=1, help='The interval between timesteps for inferring relation staes. Lower the lower memory, higher the higher accuracy.')
parser.add_argument('--t-max-see', type=int, default=49, help='How many time-steps to watch for inferring relations states')
parser.add_argument('--msg-dim', type=int, default=100, help='Dimension of message vector of relation decoder.')
parser.add_argument('--connection-value', action='store_true', default=False,
                    help='If or not to use connection value.')
parser.add_argument('--sparsity-prior', type=float, default=0.0,
                    help='Sparsity prior given to DSLR. It is valid only if connection value is used. If sparsity prior set to 0.0, sparsity prior will be not used.')
parser.add_argument('--n-relation-STD', type=int, default=5, help='The number of relation states used for relation standard deviation loss. If set to 1, relation standard deviation loss will not be used.')
parser.add_argument('--n-decoder', type=int, default=10, help='The number of time-steps of node states predicted by relation decoder in training phase.')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use.')

parser.add_argument('--RPT', action='store_true', default=False,
                    help='If set to False, we do not use reparameterization trick in relation decoder.')
parser.add_argument('--RST', action='store_true', default=False,
                    help='If set to False, we do not use random sampling trick in relation decoder.')
parser.add_argument('--video-save', action='store_true', default=False,
                    help='Generate simulation video or not.')
parser.add_argument('--edge-centrality-test-mode-random', action='store_true', default=False,
                    help='Turn on edge centrality mode (random).')
parser.add_argument('--edge-centrality-test-mode-centrality', action='store_true', default=False,
                    help='Turn on edge centrality mode (EC).')
parser.add_argument('--edge-centrality-test-mode-ratio', type=float, default=0.3,
                    help='Percent of edge that will be removed (0 - 1).')
parser.add_argument('--more-timesteps-simulation', type=int, default=0,
                    help='Predict more time steps for estimating runtime.')
parser.add_argument('--silhouette-score-test', action='store_true', default=False,
                    help='Sihlouette score test or not.')
parser.add_argument('--n-represent', type=int, default=1000,
                    help='Number of points to be represented in relation latent space.')
parser.add_argument('--cmu', action='store_true', default=False,
                    help='Test with cmu mocap data or not.')
parser.add_argument('--bball', action='store_true', default=False,
                    help='Test with basketball data or not.')

args = parser.parse_args()
args.augment = False
args.test = True

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
origin_n_f = n_f + 3 # (random_value, mass) in our paper, we didn't use mass value.
n_r_f = 128 # 300 # relation latent state의 dimension
n_fr_f = 10
msg_dim = args.msg_dim

torch.cuda.set_device(args.gpu)
t_seen = data_params['t_seen']
dt = data_params['dt']
t_interval = data_params['t_interval']
nr = data_params['nr']
t_max_see = args.t_max_see
t_seen_interval = args.t_seen_interval

if args.cmu or args.bball:
    length_of_tails = 0
else:
    length_of_tails = 75
steps = 49
if args.bball:
    steps = 8

def simulate(y):
    global ogn
    global rogn
    y = y.reshape(n, n_f).astype(np.float32)
    cur = Data(
        x=torch.from_numpy(y).cuda(),
        edge_index=e
    )
    dv = ogn.just_derivative(cur, cur.x).cpu().detach().numpy()
    y[:, dim:dim*2] = y[:, dim:dim*2] + dv * dt #* t_interval
    y[:, :dim] = y[:, :dim] + y[:, dim:dim*2] * dt #* t_interval
    return y

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _, _, _, _, _, _, data, data_y, _, _, _, _, _ = train_data_load(args, data_params)
    testloader, data_test, accel_data_test, edge_index, data_name = test_data_load(args, data_params)

    ogn, rogn = test_model_load(n_f, n_r_f, n_fr_f, msg_dim, hidden, aggr, train_name, data_name, args, data_params)

    ogn.cuda()
    rogn.cuda()

    # Loss visualization
    loss_visualization(train_name, data_name)

    # Estimate position loss, running time, and generate simulation videos
    e = edge_index.cuda()
    pos_loss = 0
    total_time = 0
    print('#test set:', len(data_test))
    for i in range( len(data_test)):
        x = np.array(data_test[i, 0, :, :])
        x = x.reshape([x.shape[0], -1, origin_n_f])[:, :, :n_f]

        if args.cmu:
            axis0, axis1, xylim = 2, 1, 1
        elif args.bball:
            axis0, axis1, xylim = 0, 1, 1
        else:
            axis0, axis1, xylim = 0, 1, 0.5

        cur_pos_loss, cur_time = video_generate_2d(train_name, data_name, i, ogn, rogn, x, e, steps, args, data_params, step_size = 1., cmu = args.cmu,
        length_of_tails = length_of_tails, video_step = 1, axis0 = axis0, axis1 = axis1, xylim = xylim, video_save = args.video_save, bball = args.bball)
        pos_loss += cur_pos_loss
        total_time += cur_time
    print('\nmean position loss: ', pos_loss/len(data_test))
    print('mean execution time: ', total_time/len(data_test))

    # Relation reasoning
    relations_training = []
    relations_test = []
    connection_values_test = []

    # Get relation states in training data
    for i in range(len(data)):
        if i> 10000: break

        rogn.before_messages = None
        x_test = np.array(data[i])[0]

        x = x_test.reshape([x_test.shape[0], -1, origin_n_f])
        x_relation = x.reshape([x.shape[0],-1])[:, -2]
        x = x[:, :, :n_f]
        x_test = x.reshape([x_test.shape[0], -1])
        x = x[:, : t_max_see:t_seen_interval, :]
        x = x.reshape([x.shape[0],-1])
        x = torch.from_numpy(x).cuda()
        x = Data(
          x=x,
          edge_index=e
        )
        rogn.just_derivative(x, x.x)

        real_r = []
        for edge in range(len(e[0])):
            r_num1 = x_relation[e[0][edge]]
            r_num2 = x_relation[e[1][edge]]
            rn = int((r_num1 * r_num2) * 10000)
            real_r.append([rn])
        relations_training.append([rogn.mean.detach(), real_r])

    # Get relation states in test data
    for i in range(len(data_test)):
        if i> 10000: break

        rogn.before_messages = None
        x_test = np.array(data_test[i])[0]

        x = x_test.reshape([x_test.shape[0], -1, origin_n_f]) # [b, t_seen, n_f]
        x_relation = x.reshape([x.shape[0],-1])[:, -2]
        x = x[:, :, :n_f] # [b, t_seen, n_f]
        x = x[:, : t_max_see:t_seen_interval, :] # [b, t_seen_random * n_f]
        x = x.reshape([x.shape[0],-1])
        x = torch.from_numpy(x).cuda()
        x = Data(
          x=x,
          edge_index=e
        )
        rogn.just_derivative(x, x.x)

        ogn.relation = rogn.relation
        x = torch.from_numpy(x_test[:, -n_f:]).cuda()
        x = Data(
          x=x,
          edge_index=e
        )
        ogn.just_derivative(x, x.x)
        real_r = []
        for edge in range(len(e[0])):
            r_num1 = x_relation[e[0][edge]]
            r_num2 = x_relation[e[1][edge]]
            rn = int((r_num1 * r_num2) * 10000)
            real_r.append([rn])
        relations_test.append([rogn.mean.detach(), real_r])

        connection_values_test.append(rogn.c.detach())

    # Pair the gt - pred of relation states
    relations_training_pair = []
    relations_test_pair = []
    connection_values_test_pair = []
    for _ in range(nr):
        relations_training_pair.append([])
        relations_test_pair.append([])
        connection_values_test_pair.append([])

    for t, m in enumerate(relations_training):
        pred = m[0]
        real = m[1]
        for i, pred_relation in enumerate(pred):
            mod_value = real[i][0] % nr
            relations_training_pair[mod_value].append(pred_relation.cpu().detach().numpy())
    for n_r in range(nr):
        relations_training_pair[n_r] = np.array(relations_training_pair[n_r])
        #print('c%d shape:'%n_r,np.shape(relations_training_pair[n_r]))

    total_relations_training = np.concatenate(relations_training_pair)
    total_relations_training = total_relations_training.astype(np.float)
    np.random.shuffle(total_relations_training)
    print('total shape (training): ', total_relations_training.shape)

    for t, m in enumerate(relations_test):
        pred = m[0]
        real = m[1]
        for i, pred_relation in enumerate(pred):
            mod_value = real[i][0] % nr
            relations_test_pair[mod_value].append(pred_relation.cpu().detach().numpy())
            connection_values_test_pair[mod_value].append(connection_values_test[t][i].cpu().detach().numpy())
    for n_r in range(nr):
        relations_test_pair[n_r] = np.array(relations_test_pair[n_r])
        connection_values_test_pair[n_r] = np.array(connection_values_test_pair[n_r])
        #print('c%d shape:'%n_r,np.shape(relations_test_pair[n_r]))

    total_relations_test = np.concatenate(relations_test_pair)
    total_relations_test = total_relations_test.astype(np.float)
    total_connection_values_test = np.concatenate(connection_values_test_pair)

    print('total shape (test): ', total_relations_test.shape)
    if np.isnan(total_relations_test).any():
        print('Nan!')
    if np.isinf(total_relations_test).any():
        print('Inf!')
    
    if args.silhouette_score_test:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        sil = []
        print('silhouette score test...')
        if data_params['sim'] in ['spring100', 'spring100_r1100']:
            for k in [2,3,4,50,100,150,200]:
              kmeans = KMeans(n_clusters = k, init="random", n_init=10, max_iter=100, random_state=6).fit(total_relations_training)
              labels = kmeans.labels_
              sil.append(silhouette_score(total_relations_training, labels, metric = 'euclidean'))
              print('k=%d silhouette score: '%k, sil[-1])
        else:
            for k in range(2, 6):
              kmeans = KMeans(n_clusters = k, init="random", n_init=10, max_iter=100, random_state=6).fit(total_relations_training)
              labels = kmeans.labels_
              sil.append(silhouette_score(total_relations_training, labels, metric = 'euclidean'))
              print('k=%d silhouette score: '%k, sil[-1])
                    
    # K-means clustering
    if nr < 5 and not args.cmu and not args.bball:
        from sklearn.cluster import KMeans
        

        if args.clustering_model == 'kmeans':
            pca = PCA(n_components=10)
            pca.fit(total_relations_training)
            model = KMeans(n_clusters=nr, init="k-means++", n_init=10, max_iter=100, random_state=6, tol = 1e-5, algorithm = 'full').fit(total_relations_training)
        elif args.clustering_model == 'GMM':
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=nr, n_init = 10, max_iter=100, random_state=6).fit(total_relations_training)

        pred_labels = []
        cur_i = 0
        right_pred =0

        idx_visited = []
        for n_r in range(nr):
            next_i = cur_i + len(relations_test_pair[n_r])
            _, label = np.unique(model.predict(total_relations_test[cur_i:next_i]), return_counts=True)
            print('connection value mean: %f'%np.mean(total_connection_values_test[cur_i:next_i]))
            pred_labels.append(label)
            cur_i = next_i
            nothing = 0
            label_ = []
            for i in range(nr):
                if i in _:
                    print('[%d]: %5d'%(_[i-nothing], label[i-nothing]), end='\t')
                    label_.append(label[i-nothing])
                else:
                    print('[%d]: %5d'%(i, 0), end='\t')
                    nothing+=1
                    label_.append(0)
            print()
            for idx in idx_visited:
                label_[idx] = 0
            label_ = np.array(label_)
            right_pred += label_.max()
            idx_visited.append(label_.argmax())
        print( right_pred / len(total_relations_test) * 100 , '%')

    # PCA visualization
    colors = ['red', 'blue', 'green', 'orange','black','purple','cyan']
    pca = PCA(n_components=2)
    pca.fit(total_relations_training)
    pca_c = pca.transform(total_relations_test)
    cur_i = 0
    plt.clf()
    plt.figure(figsize=(5, 5))
    
    total_connection_values_test = 1- total_connection_values_test
    total_connection_values_test = total_connection_values_test.reshape([-1,1])
    #total_connection_values_test = np.log(total_connection_values_test + 1e-4)
    for n_r in range(nr):
        next_i = cur_i + len(relations_test_pair[n_r])
        if args.bball:
            
            total_connection_values_test = (total_connection_values_test - total_connection_values_test.min())/(total_connection_values_test.max() - total_connection_values_test.min())
            cs_color = np.concatenate([total_connection_values_test, total_connection_values_test*0, 1-total_connection_values_test], axis=-1)
            alphas = (total_connection_values_test > 0.5) *0.03 + 0.01
            scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= cs_color[:args.n_represent], s= 5, alpha = alphas[:args.n_represent])
        
        elif args.cmu:
            total_connection_values_test = (total_connection_values_test - total_connection_values_test.min())/(total_connection_values_test.max() - total_connection_values_test.min())
            cs_color = np.concatenate([total_connection_values_test, total_connection_values_test*0, 1-total_connection_values_test], axis=-1)
            #cs_color = np.concatenate([total_connection_values_test**0.3, total_connection_values_test*0, 1-total_connection_values_test**0.3], axis=-1)
            print(cs_color)
            alphas = (total_connection_values_test > 0.5) * 0.03 + 0.01
            scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= cs_color[:args.n_represent], s= 1, alpha = alphas[:args.n_represent])
        elif data_params['sim'] == 'spring100':
            scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= [[1. * n_r / 100, 0, 1. * (100 - n_r) / 100]]*min(next_i - cur_i, args.n_represent), s= 1,
            alpha = 0.1)
        elif data_params['sim'] == 'spring100_r1100':
            scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent],
            c= [[1. * n_r * (n_r < 100) / 100,  1. * (n_r - 100) * (n_r >= 100) / 100, 1. - 1. * n_r * (n_r < 100) / 100 - 1. * (n_r - 100) * (n_r >= 100) / 100,]]*min(next_i - cur_i, args.n_represent), s= 1,
            alpha = 0.1)
        else:
            scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= colors[n_r][:args.n_represent], s= 0.1)
        cur_i = next_i
    plt.xlabel('X')
    plt.ylabel('Y')
    relation_fig_save(plt, train_name, data_name)
    
    if args.connection_value:
        cur_i = 0
        plt.clf()
        plt.figure(figsize=(3, 3))
        #total_connection_values_test = (total_connection_values_test - total_connection_values_test.min())/(total_connection_values_test.max() - total_connection_values_test.min())
        cs_color = np.concatenate([total_connection_values_test, total_connection_values_test*0, 1-total_connection_values_test], axis=-1)
        for n_r in range(nr):
            next_i = cur_i + len(relations_test_pair[n_r])
            print('평균 중요도:',total_connection_values_test[cur_i:next_i].mean())
            if args.cmu or args.bball:

                #total_connection_values_test = (total_connection_values_test - total_connection_values_test.min())/(total_connection_values_test.max() - total_connection_values_test.min())
                print(total_connection_values_test)
                print(total_connection_values_test.mean())
                cs_color = np.concatenate([total_connection_values_test**0.1, total_connection_values_test*0, 1-total_connection_values_test**0.1], axis=-1)
                cs_color = np.concatenate([total_connection_values_test, total_connection_values_test*0, 1-total_connection_values_test], axis=-1)
                alphas = (total_connection_values_test > 0.5) *0.03 + 0.01
                scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= cs_color[:args.n_represent], s= 5, alpha = alphas[:args.n_represent])
            elif data_params['sim'] == 'spring100':
                scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= cs_color[cur_i:next_i][:args.n_represent], s= 1, alpha = 0.01)
            elif data_params['sim'] == 'spring100_r1100':
                scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent],
                c= cs_color[cur_i:next_i][:args.n_represent], s= 1, alpha = 0.1)
            else:
                scatter = plt.scatter(pca_c[cur_i:next_i,0][:args.n_represent],pca_c[cur_i:next_i,1][:args.n_represent], c= cs_color[cur_i:next_i][:args.n_represent], s= 0.1, alpha = 0.01)
                #scatter = plt.scatter(pca_c[cur_i:next_i,0]
            cur_i = next_i
        plt.xlabel('X')
        plt.ylabel('Y')
        relation_fig_save(plt, train_name, data_name, suffix='_rc')

if __name__ == '__main__':
    main()
