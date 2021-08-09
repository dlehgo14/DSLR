import jax
from jax import ops
from jax.ops import index_update
from jax import numpy as np
from matplotlib import pyplot as plt
from jax import jit, vmap, grad, pmap
from jax.experimental.ode import odeint
from jax import random
import numpy as onp
import matplotlib as mpl
import multiprocessing
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from celluloid import Camera
s_ = onp.s_
tqdm = lambda _: _

def make_transparent_color(ntimes, fraction):
  rgba = onp.ones((ntimes, 4))
  alpha = onp.linspace(0, 1, ntimes)[:, np.newaxis]
  color = np.array(mpl.colors.to_rgba(mpl.cm.gist_ncar(fraction)))[np.newaxis, :]
  rgba[:, :] = 1*(1-alpha) + color*alpha
  rgba[:, 3] = alpha[:, 0]
  return rgba

def get_potential(sim, sim_obj):

    dim = sim_obj._dim

    @jit
    def potential(x1, x2):
      """The potential between nodes x1 and x2"""
      dist = np.sqrt(np.sum(np.square(x1[:dim] - x2[:dim])))
      #Prevent singularities:

      min_dist = 1e-2
      bounded_dist = dist + min_dist
    #   test_dist(jax.device_get(np.sum(np.any(dist <= min_dist))))
      rn = ((x1[-2] * x2[-2]) * 10000).astype(int)

      if sim == 'spring_None':
          potential = (bounded_dist - 1)**2 * (rn % 2 == 0)
          return potential
      elif sim == 'r1_None':
          potential = x1[-1]*x2[-1]*np.log(bounded_dist) * (rn % 2 == 0)
          return potential
      elif sim == 'spring_r1':
          potential = x1[-1]*x2[-1]*np.log(bounded_dist) * (rn % 2 == 0) + \
                        (bounded_dist - 1)**2 * (rn % 2 == 1)
          return potential
      elif sim == 'spring_r1_None':
          potential = (bounded_dist - 1)**2 * (rn % 3 == 0) + \
                        x1[-1]*x2[-1]*np.log(bounded_dist) * (rn % 3 == 1)
          return potential
      elif sim == 'spring4':
          potential = (bounded_dist - 1)**2 * (rn % 4) / 4
          return potential
      elif sim == 'r14':
          potential = x1[-1]*x2[-1]*np.log(bounded_dist) * (rn % 4) / 2
          return potential
      elif sim == 'spring100':
          potential = (bounded_dist - 1)**2 * (rn % 100) * 0.03
          #potential = (bounded_dist - 1)**2 * ( (rn % 100 / 25).astype(int) * 25) * 0.03
          #potential = (bounded_dist - 1)**2 * (rn % 3 == 0) + \
          #              x1[-1]*x2[-1]*np.log(bounded_dist) * (rn % 3 == 1)
          return potential
      elif sim == 'spring100_r1100':
          r_value = rn % 200
          potential = (bounded_dist - 1)**2 * r_value * 0.03 * (r_value < 100) + \
                    x1[-1]*x2[-1]*np.log(bounded_dist) * ( (r_value - 100) * 0.03)  * (r_value >= 100)
          return potential
      else:
          raise NotImplementedError('No such simulation ' + str(sim))

    return potential

class SimulationDataset(object):

    """Docstring for SimulationDataset. """

    def __init__(self, sim='r2', n=5, dim=2,
            dt=0.01, nt=100, extra_potential=None,
            **kwargs):
        """TODO: to be defined.

        :sim: Simulation to run
        :n: number of bodies
        :nt: number of timesteps returned
        :dt: time step (can also set self.times later)
        :dim: dimension of simulation
        :pairwise: custom pairwise potential taking two nodes as arguments
        :extra_potential: function taking a single node, giving a potential
        :kwargs: other kwargs for sim

        """
        self._sim = sim
        self._n = n
        self._dim = dim
        self._kwargs = kwargs
        self.dt = dt
        self.nt = nt
        self.data = None
        self.times = np.linspace(0, self.dt*self.nt, num=self.nt)
        self.G = 1
        self.extra_potential = extra_potential
        self.pairwise = get_potential(sim=sim, sim_obj=self)

    def simulate(self, ns, key=0):
        rng = random.PRNGKey(key)
        vp = jit(vmap(self.pairwise, (None, 0), 0))
        n = self._n # number of nodes
        dim = self._dim
        sim = self._sim
        params = 3 # random_value, mass
        total_dim = dim*2+params
        times = self.times
        G = self.G
        if self.extra_potential is not None:
          vex = vmap(self.extra_potential, 0, 0)

        @jit
        def total_potential(xt):
          sum_potential = np.zeros(())
          for i in range(n - 1):
            sum_potential = sum_potential + G*vp(xt[i], xt[i+1:]).sum()
          print(sum_potential)
          return sum_potential

        @jit
        def force(xt):
          return -grad(total_potential)(xt)[:, :dim]

        @jit
        def acceleration(xt):
          return force(xt)/xt[:, -1, np.newaxis]

        unpacked_shape = (n, total_dim)
        packed_shape = n*total_dim

        @jit
        def odefunc(y, t):
          dim = self._dim
          y = y.reshape(unpacked_shape)
          a = acceleration(y)
          return np.concatenate(
              [y[:, dim:2*dim],
               a, 0.0*y[:, :params]], axis=1).reshape(packed_shape)

        @partial(jit, backend='cpu')
        def make_sim(key):
            x0 = random.normal(key, (n, total_dim)) # (n, dim * 2 + params)
            #x0 = index_update(x0, s_[..., -1], np.exp(x0[..., -1])) #all masses
            x0 = index_update(x0, s_[..., -1], 1) # mass set to 1, [ x, y, x', y', random_value, mass]
            x_times = odeint(
                odefunc,
                x0.reshape(packed_shape),
                times, mxstep=2000).reshape(-1, *unpacked_shape)
            return x_times

        keys = random.split(rng, ns)
        #vmake_sim = jit(vmap(make_sim, 0, 0), backend='cpu')
        # self.data = jax.device_get(vmake_sim(keys))
        # self.data = np.concatenate([jax.device_get(make_sim(key)) for key in keys])
        data = []
        for key in tqdm(keys):
            data.append(make_sim(key))
        self.data = np.array(data)

    def get_acceleration(self):
        vp = jit(vmap(self.pairwise, (None, 0), 0))
        n = self._n
        dim = self._dim
        sim = self._sim
        params = 3
        total_dim = dim*2+params
        times = self.times
        G = self.G
        if self.extra_potential is not None:
          vex = vmap(self.extra_potential, 0, 0)
        @jit
        def total_potential(xt):
          sum_potential = np.zeros(())
          for i in range(n - 1):
            sum_potential = sum_potential + G*vp(xt[i], xt[i+1:]).sum()
          if self.extra_potential is not None:
            sum_potential = sum_potential + vex(xt).sum()
          return sum_potential

        @jit
        def force(xt):
          return -grad(total_potential)(xt)[:, :dim]

        @jit
        def acceleration(xt):
          return force(xt)/xt[:, -1, np.newaxis]

        vacc = vmap(acceleration, 0, 0)
        # ^ over time
        vacc2 = vmap(vacc, 0, 0)
        # ^ over batch
        return vacc2(self.data)

    def plot(self, i, animate=False, plot_size=True, s_size=1):
        #Plots i
        n = self._n
        times = onp.array(self.times)
        x_times = onp.array(self.data[i])
        sim = self._sim
        masses = x_times[:, :, -1]
        if not animate:
            for j in range(n):
              rgba = make_transparent_color(len(times), j/n)
              if plot_size:
                plt.scatter(x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=3*masses[:, j]*s_size)
              else:
                plt.scatter(x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=s_size)
        else:
            fig = plt.figure()
            camera = Camera(fig)
            d_idx = 20
            for t_idx in range(d_idx, len(times), d_idx):
                start = max([0, t_idx-300])
                ctimes = times[start:t_idx]
                cx_times = x_times[start:t_idx]
                for j in range(n):
                  rgba = make_transparent_color(len(ctimes), j/n)
                  if plot_size:
                    plt.scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=3*masses[:, j])
                  else:
                    plt.scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=s_size)
#                 plt.xlim(-10, 10)
#                 plt.ylim(-10, 10)
                camera.snap()
            from IPython.display import HTML
            return HTML(camera.animate().to_jshtml())
