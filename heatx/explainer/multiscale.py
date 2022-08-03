import jax.numpy as jnp
from jax import device_put
import jax 
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import gzip
from torchvision.utils import save_image
from torch.autograd import Variable
import cv2
import jax.lax as lax
from functools import partial
from jax import random, vmap
from .model_flax import restore_checkpoints, create_learning_rate_fn, train_step
import optax
from flax.training import checkpoints

def compress(filepath, arr):
    arr = (arr*255).astype(np.uint8)
    f = gzip.GzipFile(filepath, "w")
    np.save(f, arr)
    f.close()

def load(filepath):
    f = gzip.GzipFile(filepath, "r")
    arr = np.load(f)
    return arr/255.

class MultiScale(object):
    def __init__(self, input_shape,
                       netclass,
                       manifold_latent,
                       data_encode,
                       VAE_model,
                       params,
                       regression,
                       save_path,
                       stepsize = 0.01,
                       steps = 5000,
                       skip = 10,
                       walk_bs = 200,
                       device='cpu'):
        
        self.save_path = save_path
        self.input_shape = input_shape
        self.D = np.prod([k for k in input_shape])
        self.netclass = netclass
        self.manifold_latent = manifold_latent
        self.regression = regression

        if data_encode is not None:
            self.data_encode = device_put(data_encode) 
        self.VAE_model = VAE_model
        self.params = params
        self.stepsize = stepsize
        self.steps = steps
        self.skip = skip
        self.walk_bs = walk_bs
        self.device = device

    @partial(jax.jit, static_argnums=(0,), static_argnames=('num_steps',))
    def random_walk(self, prng, z, *, step_size=0.01, num_steps=10):
        def body(carry, _):
            z_, key = carry
            key, subkey = random.split(key)
            noise = random.normal(subkey, z_.shape)
            M = self.manifold_latent.metric_tensor_flax(z_)
            L,U = vmap(jnp.linalg.eigh)(M)
            coef = vmap(jnp.dot)(U, 1./jnp.sqrt(L) *noise)
            zt = z_ + step_size * coef
            carry = (zt, key)
            return carry, zt

        prefix, dim_z = z.shape[:-1], z.shape[-1]
        z0 = jnp.reshape(z, (-1, dim_z))
        _, zts = lax.scan(body, (z0, prng), jnp.arange(num_steps))
        zts = jnp.reshape(zts, (num_steps, *prefix, dim_z))
        return zts
 
    
    def solve_heat_kernel(self, N_train, size=None, trainbs=32, epochs=20):
        """
        Given a black-box model to explain, generate samples with random walk as input

        Args:
            model: black-box model to explain, can evaluate at any point
            N_train: number of training samples for each time step 
        """

        Z_start = self.data_encode[np.random.randint(0, self.data_encode.shape[0], N_train)]
        times = int(self.steps / self.skip)
        print("Total time to process: ", times)

        model_state = restore_checkpoints(self.netclass, self.netclass.modeldir)
        heatkernel_state = restore_checkpoints(self.netclass, self.netclass.modeldir)

        base_learning_rate = self.netclass.learning_rate * trainbs / 256.
        steps_per_epoch = N_train // trainbs

        randomidx = np.random.choice(range(N_train),5)
        z_sample = Z_start[randomidx,:]
        for k in tqdm(range(times)):
            x_sample = self.VAE_model.apply(self.params, z_sample, mode='decode') 
            im = []
            for j in range(x_sample.shape[0]):
                sample = np.asarray(x_sample[j]*255).astype(np.int32)
                sample = np.swapaxes(np.swapaxes(sample,0,1),1,2)
                im.append(sample)
            im = np.concatenate(im)
            cv2.imwrite(str(self.save_path/'generated_random_samples'/ ('network_step' + str(k) + '.png')),im)
            key = random.PRNGKey(k)
            z_sample = self.random_walk(key, z_sample, step_size=self.stepsize, num_steps=self.skip)[-1]



        Z_old = Z_start.copy()
        for k in range(1, times):
            learning_rate_fn = create_learning_rate_fn(
                    self.netclass, base_learning_rate, steps_per_epoch)
            tx = optax.sgd(
                learning_rate=learning_rate_fn,
                momentum=self.netclass.momentum,
                nesterov=True,
            )
            heatkernel_state.tx = tx


            Z_step = jnp.empty((0, Z_start.shape[1]))
            correct = 0
            for i,batch_idx in enumerate(range(0,N_train,trainbs)):
                indices = np.arange(N_train)[batch_idx:batch_idx+trainbs]
                key = random.PRNGKey(k)
                z_step = self.random_walk(key, Z_old[indices,:],step_size=self.stepsize, num_steps=self.skip)[-1]
                z_start = Z_start[indices,:]
                Z_step = jnp.concatenate([Z_step, z_step])
                
                X_batch = self.VAE_model.apply(self.params, z_step, mode='decode')
                X_start = self.VAE_model.apply(self.params, z_start, mode='decode')
                
                if i==0:
                    if not (self.save_path/'generated_samples').is_dir(): os.mkdir(self.save_path/'generated_samples')
                    save_image(X_batch, self.save_path/'generated_samples'/ ('network_step' + str(k) + '.png'))

                y_batch = model_state.apply_fn(model_state.params, device_put(X_batch))
                batch = {'image': device_put(X_start), 'label': y_batch}
                heatkernel_state, metrics = train_step(heatkernel_state, batch, learning_rate_fn, loss='mse')

                if i % 30 == 0:
                        print('Train Epoch: {} [\t{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                0, i * trainbs, N_train,
                                100. * i * trainbs / N_train, metrics['loss']))

            for epoch in range(1,epochs):
                permutation = np.random.permutation(N_train)
                for i,batch_idx in enumerate(range(0,N_train,trainbs)):
                    indices = permutation[batch_idx:batch_idx + trainbs]
                    z_step = Z_step[indices,:]
                    z_start = Z_start[indices,:]

                    X_batch = self.VAE_model.apply(self.params, z_step, mode='decode')
                    X_start = self.VAE_model.apply(self.params, z_start, mode='decode')
                    y_batch = model_state.apply_fn(model_state.params, device_put(X_batch))
                    batch = {'image': device_put(X_start), 'label': y_batch}
                    heatkernel_state, metrics = train_step(heatkernel_state, batch, learning_rate_fn, loss='mse')

                    if i % 30 == 0:
                            print('Train Epoch: {} [\t{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, i * trainbs, N_train,
                                    100. * i * trainbs / N_train, metrics['loss']))
                
            if not (self.save_path / 'explain_model').is_dir():
                os.mkdir(self.save_path / 'explain_model')
            checkpoints.save_checkpoint(str(self.save_path / 'explain_model'), 
                                        target=heatkernel_state, 
                                        step=k, 
                                        prefix='time_', keep=times, overwrite=False)
            print("Finished training for timestep ", str(k))

            Z_old = Z_step

            
    def palpha_square(self, data, index, stop, size, shape, skip=100, accum=True):
        '''
            decompose heatkernel h(x,T) - f(x) by each feature
            = \sum_i \int_0^T P_i^2 h(x,t) dt = \sum_i \sum_k P_i^2 h(x, t_k) delta_t (left sum)

            Args:
                index: which output to explain
                start: h(x, start*delta)
                stop: h(x, stop*delta)

            Return:
                results: accumulated attribution for each feature at timet, # t x D
                prediction: output of h(x,t) at each t, # t+1
        '''
        delta = self.stepsize * self.skip
        result, results = np.empty((0, self.D)),np.zeros((self.D,),dtype=np.float32)
        prediction = []

        _,encodez,_ = self.VAE_model.apply(self.vae_params,data[None,:])
        d = encodez.shape[-1]
        decoder = lambda z: self.VAE_model.apply(self.vae_params, z, mode='decode')
        Jg = jnp.squeeze(jax.jacfwd(decoder)(encodez)).reshape(-1,d)
        Hg = jnp.squeeze(jax.jacfwd(jax.jacfwd(decoder))(encodez)).reshape(-1,d,d)

        Ginv = jnp.linalg.inv(Jg.T.dot(Jg)) # d,d
        JgGinv = Jg.dot(Ginv)

        for k in tqdm(range(0,stop)):
            path = str(self.save_path/'explain_model') if k>0 else str(self.save_path/'network')
            heatkernel_state = restore_checkpoints(self.netclass, path)
            output = heatkernel_state.apply_fn(heatkernel_state.params, data/255.)

            def ztoy(z):
                x = self.VAE_model.apply(self.vae_params, z, mode='decode')
                x = jnp.transpose(x, (0,2,3,1))
                y = heatkernel_state.apply_fn(heatkernel_state.params, x)
                return y

            gradf = jnp.squeeze(jax.jacfwd(ztoy)(encodez))
            Hf = jnp.squeeze(jax.jacfwd(jax.jacfwd(ztoy))(encodez))

            HJGinvf = jnp.einsum('kab,k->ab', Hg, JgGinv.dot(gradf))
            JHGinvf = Jg.T.dot(Hg.dot(Ginv).dot(gradf))
            ## Pa(Paf)
            #h1 = Hg.dot(Ginv).dot(gradf) - JgGinv.dot(HJGinvf + JHGinvf - Hf)
            ## Hess f(Pa, Pa)
            h2 = JgGinv.dot(Hf - HJGinvf)

            #result1_k = np.einsum('kd,kd->k', JgGinv, h1)
            result2_k = np.einsum('kd,kd->k', JgGinv, h2)

            ## Pa(Paf) + div(Pa)Paf
            #result3_k = result1_k + \
            #    (- np.sum((np.diagonal(Hg,axis1=1,axis2=2).T).dot(Jg).dot(Ginv.dot(Jg.T)),axis=0) \
            #        + np.sum(np.diagonal(Hg,axis1=1,axis2=2),axis=1)) * Jg.dot(Ginv).dot(gradf)


            result += delta * result2_k
            if k % skip == 0:
                if k >0 : results = np.concatenate([results, result[None,:]])
                prediction.append(output)
                if not accum:
                    result = np.zeros((self.D,),dtype=np.float32)
        return results, prediction

    def explain_multiscale(self, data, projection=False, skip=10):
        """
        explain new test point

        Return:
            # shape = (timesteps, number of outputs, *input shape)
        """

        if len(data.shape)!=4:
            data = data.reshape(-1,*self.input_shape)
        x = Variable(data, requires_grad=True)

        gradient = np.empty((0,*self.input_shape))
        if projection:
            encodez = self.VAE_model.encode(data.view(-1, self.D))[0].detach().numpy()
            proj = self.manifold_latent.project(encodez.T)
            projection = True
        
        gradient = np.empty((0,0,*self.input_shape))
        times = max([int(f[:-4]) for f in os.listdir(self.save_path/'explain_model') if f[-4:]=='.pth'])
        for k in range(0,times, skip):
            heatkernel = self.netclass(size)
            path = self.save_path/'explain_model'/(str(k)+'.pth') if k>0 else self.save_path / 'network.pth'
            heatkernel.load_state_dict(torch.load(path)['model_stats'])
            heatkernel.eval()

            x = Variable(data, requires_grad=True)
            output = heatkernel(x)

            grad_k = np.empty((0,*self.input_shape))
            for index in range(output.shape[-1]):
                one_hot = np.zeros((1, output.size()[-1]),dtype=np.float32)
                one_hot[0][index]=1
                one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
                one_hot = torch.sum(one_hot * output)

                if x.grad is not None:
                    x.grad.data.zero_()   
                one_hot.backward(retain_graph=True)
                grad = (x.grad.data.view(self.D,).cpu().numpy())
                if projection: grad = proj(grad).reshape(self.input_shape)
                grad_k = np.concatenate([grad_k, grad[None,:]], axis=0)

            gradient = np.concatenate([gradient, grad_k[None,:]], axis=0)

            return gradient
