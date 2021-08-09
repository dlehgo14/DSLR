from src.simulate import *
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

sim_list = {
  'spring_None':2,
  'r1_None':2,
  'spring_r1':2,
  'spring_r1_None':3,
  'spring4':4,
  'r14':4,
  'spring100':100,
  'spring100_r1100':200,
}

# dataset parameters
parser = argparse.ArgumentParser()
parser.add_argument('--t-seen', type=int, default=99, help='Time steps per data')
parser.add_argument('--t-interval', type=int, default=5, help='Time interval per step')
parser.add_argument('--ns', type=int, default=50000, help='Number of simulations to run')
parser.add_argument('--n', type=int, default=5, help='Number of nodes')
parser.add_argument('--dim', type=int, default=2, help='Dimension of simulation')
parser.add_argument('--sim', type=str, default='spring_None', help='Simulation type to generate. Check the available simulation lists in sim_list in data_generator.py.')
args = parser.parse_args()

def main():
    assert args.sim in sim_list
    # Number of time steps
    nt = args.t_seen * 5
    #Select the hand-tuned dt value for a smooth simulation
    # (since scales are different in each potential):
    dt = 5e-3
    title = '{}_n={}_dim={}_nt={}_dt={}'.format(args.sim, args.n, args.dim, nt, dt)
    print('Running on', title)

    s = SimulationDataset(args.sim, n=args.n, dim=args.dim, nt=nt, dt=dt)
    s.simulate(args.ns)
    data_ = s.data

    s_test = SimulationDataset(args.sim, n=args.n, dim=args.dim, nt=nt, dt=dt)
    s_test.simulate(500, key = 1)
    data_test_ = s_test.data

    s_valid = SimulationDataset(args.sim, n=args.n, dim=args.dim, nt=nt, dt=dt)
    s_valid.simulate(500, key = 2)
    data_valid_ = s_valid.data

    # Generate training data:
    data = []
    delete_i = []
    for i, d in enumerate(data_):
      n_nan = np.count_nonzero(np.isnan(d))
      if n_nan != 0:
        delete_i.append(i)

    for k, data_sequence in enumerate(data_):
      if k in delete_i:
        continue
      data.append([])
      # [ nt, n, n_f]
      for i, data_instant in enumerate(data_sequence):
        # [n, n_f]
        if i < args.t_seen * args.t_interval - 1:
          continue
        data_continuous=np.concatenate([data_sequence[t] for t in range(i-args.t_seen*args.t_interval+1, i+1, args.t_interval)],
                                         axis=-1)
        data[-1].append(data_continuous)
      if (k+1) % 100 == 0:
        print('\r%7d/%7d'%(k+1,len(data_)),end='')
    data = np.array(data)
    print()

    # Generate test data:
    data_test = []
    delete_i = []
    for i, d in enumerate(data_test_):
      n_nan = np.count_nonzero(np.isnan(d))
      if n_nan != 0:
        delete_i.append(i)

    for k, data_sequence in enumerate(data_test_):
      if k in delete_i:
        continue
      data_test.append([0])
      # [ nt//2, n, n_f]
      for i, data_instant in enumerate(data_sequence):
        # [n, n_f]
        if i < args.t_seen * args.t_interval - 1:
          continue
        data_continuous=np.concatenate([data_sequence[t] for t in range(i-args.t_seen*args.t_interval+1, i+1, args.t_interval)],
                                       axis=-1)
        data_test[-1][0] = data_continuous
      if (k+1) % 100 == 0:
        print('\r%7d/%7d'%(k+1,len(data_test_)),end='')
    data_test = np.array(data_test)
    print()

    # Generate valid data:
    data_valid = []
    delete_i = []
    for i, d in enumerate(data_valid_):
      n_nan = np.count_nonzero(np.isnan(d))
      if n_nan != 0:
        delete_i.append(i)

    for k, data_sequence in enumerate(data_valid_):
      if k in delete_i:
        continue
      data_valid.append([0])
      # [ nt//2, n, n_f]
      for i, data_instant in enumerate(data_sequence):
        # [n, n_f]
        if i < args.t_seen * args.t_interval - 1:
          continue
        data_continuous=np.concatenate([data_sequence[t] for t in range(i-args.t_seen*args.t_interval+1, i+1, args.t_interval)],
                                       axis=-1)
        data_valid[-1][0] = data_continuous
      if (k+1) % 100 == 0:
        print('\r%7d/%7d'%(k+1,len(data_valid_)),end='')
    data_valid = np.array(data_valid)
    print()

    data_name = "r%d_o%d_tseen%d_ns%d_dim%d_nt%d_dt%.3f_%s"%(sim_list[args.sim],args.n,args.t_seen,args.ns,args.dim, nt, dt, args.sim)
    np.savez('./data/%s.npz'%data_name, data=data, data_test=data_test, data_valid=data_valid)
    print('%s save completed.'%data_name)

if __name__ == '__main__':
    main()
