import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import copy
import time

#from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from scipy.stats import norm
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.stats import pearsonr

# from src.bound.compute_bounds import *
from pynvml import *

from src.bound.compute_bounds import *


def print_gpu_utilization():
    nvmlInit()
    memory = 0
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    # print(info.used, info.total)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    memory += info.used//1024**2

class NNHessianCalculator():
    def __init__(self, model, loss_fn, dataloader=None, external_load_batch_func=None, assigned_parameters=[], device='cpu'):
        self.model = model.eval()  # make model is in evaluation model
        self.loss_fn = loss_fn
        self.aggregate_method = 'mean'

        if external_load_batch_func is not None:
            self.load_batch_func = external_load_batch_func
        else:
            self.load_batch_func = load_batch_func
        
        self.dataloader = dataloader
        self.train_values = None
        self.train_weights = None
        self.device = device
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    ##############################################
    # Utils
    ##############################################

    def _hessian_vector_product(self, v=None):
        if v is None:
            v = torch.randn(self.total_params, device=self.device)
        v = v.to(self.device)
        self.model.zero_grad()
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'
        #if dataloader is None:
        #    dataloader = self.dataloader
        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch in dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            if self.lino_sigma == 0:
                output = self.model(data)
            else:
                output = self.model(data, embed_noise=True, sigma=self.lino_sigma, method='rule_noise')
            loss = self.loss_fn(output, target, 'mean')
            #self.model.zero_grad()

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor * batch_size

        total_hd_tensor /= len(dataloader.dataset)
            #if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
            #    break
        return total_hd_tensor

    def evaluate_loss(self, model, dataloader, loss_fn, device):
        """Compute the average loss over a dataloader."""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        return total_loss / total_samples
    
    def evaluate_hutch(self, model, dataloader, loss_fn, noise_vector=None, device='cpu'):
        model.train()
        total_loss = 0.0
        total_hutch = 0.0
        total_max_eig = 0.0
        count = 0
        with sdpa_kernel(SDPBackend.MATH):
            for batch in dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * batch_size
                count += batch_size
                
                # Generate a noise vector matching the flattened parameters.
                if noise_vector is None:
                    flat_params = torch.cat([p.detach().view(-1) for p in model.parameters()])
                    noise_vector = torch.randn_like(flat_params)
                
                # Hutchinson estimator: v^T H v
                hessian_quad = self.hessian_quadratic_form(model, loss, noise_vector)
                hessian_quad = hessian_quad.item()
                total_hutch += hessian_quad * batch_size

        avg_loss = total_loss / count if count > 0 else 0.0
        avg_hutch = total_hutch / count if count > 0 else 0.0
        return avg_loss, avg_hutch
    
    def evaluate_max_eigenvalue(self, model, dataloader, loss_fn, device='cpu'):
        model.train()
        total_max_eig = 0.0
        count = 0
        
        with sdpa_kernel(SDPBackend.MATH):
            for batch in dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target)
                count += batch_size
                
                # Compute the largest eigenvalue of the Hessian.
                max_eig = self.approximate_lambda_max(loss, model, power_iter=100)
                total_max_eig += max_eig * batch_size

        avg_max_eig = total_max_eig / count if count > 0 else 0.0
        return avg_max_eig

    
    ##############################################
    # Main algorithm 1: stochastic lanczos quadrature
    ##############################################

    def get_train_spectrum(self, n_v, n_iter):
        if self.train_weights is None or self.train_values is None:
            self.train_values, self.train_weights = self.get_full_spectrum(n_v, n_iter, self.dataloader)
        return self.train_values, self.train_weights
    
    def get_valid_spectrum(self, n_v, n_iter):
        if self.valid_weights is None or self.valid_values is None:
            self.valid_values, self.valid_weights = self.get_full_spectrum(n_v, n_iter, self.valid_dataloader)
        return self.valid_values, self.valid_weights

    def get_full_spectrum(self, n_v, n_iter, dataloader=None):
        weights = np.zeros((n_v, n_iter))
        values = np.zeros((n_v, n_iter))

        for k in range(n_v): 
            'wiki version'
            T = self.tridiagonalize_by_lanzcos(n_iter, k, dataloader)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
            if k == 0:
                print_gpu_utilization()
        
        all_values = np.concatenate(values)
        all_weights = np.concatenate(weights)
        return all_values, all_weights
   
        grid, curve = self.interpolate(weights, values)
    
    def tridiagonalize_by_lanzcos(self, n_iter, k, dataloader=None):
        'set up'
        v_list = []
        T = np.zeros((n_iter, n_iter), dtype= np.float64)

        'initialization'
        v = torch.randn(self.total_params, dtype = torch.float64) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], dataloader)
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        #print('runing lanczos')
        for j in range(1, n_iter):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory


            w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], dataloader)
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta
         
        return  T

    def hessian_vector_product_with_tensor_input(self, d_tensor, dataloader=None):
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'
        #if dataloader is None:
        #    dataloader = self.dataloader
        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch in dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            if self.lino_sigma == 0:
                output = self.model(data)
            else:
                output = self.model(data, embed_noise=True, sigma=self.lino_sigma, method='rule_noise')
            loss = self.loss_fn(output, target, 'mean')
            #self.model.zero_grad()

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor * batch_size

        total_hd_tensor /= len(dataloader.dataset)
            #if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
            #    break
        return total_hd_tensor

    ##############################################
    # Main algorithm 2: Hutchinson’s Method
    ##############################################

    def approximate_lambda_max(self, loss, model, power_iter=20):
        """
        Approximates the largest eigenvalue of the Hessian with respect to 'weights'
        using power iteration.
        Only works well if the Hessian is PSD.
        
        Returns: float lambda_max
        """
        # Compute first-order gradient with create_graph=True for higher-order derivatives.
        gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in gradients])
        
        # Initialize a random vector v of same shape as grad_vector
        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)
        
        for _ in range(power_iter):
            # Compute dot product between grad_vector and v (a scalar)
            dot = torch.dot(grad_vector, v)
            # Compute Hessian-vector product; allow_unused=True avoids error if some weights don't affect dot.
            Hv_tuple = torch.autograd.grad(dot, model.parameters(), retain_graph=True, allow_unused=True)
            # Replace any None gradients with zero tensors of the same shape
            Hv_list = []
            for idx, h in enumerate(Hv_tuple):
                Hv_list.append(h)
            # Flatten the gradients to a single vector
            Hv = torch.cat([h.reshape(-1) for h in Hv_list])
            norm_Hv = torch.norm(Hv, p=2)
            if norm_Hv < 1e-8:
                return 0.0
            v = Hv / norm_Hv

        # Final Rayleigh quotient approximation for the eigenvalue
        #final_dot = torch.dot(grad_vector, v)
        #lambda_max_approx = final_dot.item()
        lambda_max_approx = norm_Hv.item()
        return lambda_max_approx

    def hessian_quadratic_form(self, model, loss, noise_vector):
        
        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])
        
        # Compute the dot product between gradients and noise vector.
        grad_dot_noise = torch.dot(grad_vector, noise_vector)
        
        # Compute Hessian-vector product using the Pearlmutter trick.
        Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
        Hv_vector = torch.cat([h.reshape(-1) for h in Hv])
        
        # The quadratic form δ^T H δ.
        quad_form = torch.dot(noise_vector, Hv_vector)
        return quad_form

    def hutch_pp_trace_estimator(self, model, loss, m):
        """
        Estimate the trace of the Hessian (i.e., tr(H)) using the Hutch++ algorithm.
        
        Args:
            model: A torch.nn.Module whose parameters are used.
            loss: A scalar loss computed from the model.
            m: The total number of Hessian–vector queries. Must be chosen so that
            s = (m+2)//4 and g = (m-2)//2 are integers.
        
        Returns:
            trace_estimate: A scalar tensor estimating the trace of the Hessian.
        """
        # Compute total number of parameters d.
        params = list(model.parameters())
        d = sum(p.numel() for p in params)
        
        # Set the number of columns in S and G.
        s = (m + 2) // 4        # number of queries for the projection subspace
        g_num = (m - 2) // 2      # number of queries for the residual term
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Sample S ~ N(0,1) in R^(d x s)
        S = torch.randn(d, s, device=device, dtype=dtype)
        
        # Sample G with Rademacher entries in R^(d x g_num): values are +1 or -1.
        # torch.randint returns integers; we then cast them to the appropriate type.
        G = torch.randint(0, 2, (d, g_num), device=device).float() * 2 - 1
        G = G.to(dtype)
        
        # Define a helper function to compute the Hessian-vector product (H*v)
        def hvp(v):
            # Compute gradients (first derivative) with create_graph=True to allow second-order derivatives.
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grad_vector = torch.cat([g.reshape(-1) for g in grads])
            # Compute the directional derivative of grad_vector in direction v.
            dot = torch.dot(grad_vector, v)
            hv = torch.autograd.grad(dot, model.parameters(), retain_graph=True)
            hv_vector = torch.cat([h.reshape(-1) for h in hv])
            return hv_vector

        # Compute Y = A*S, where each column i is hvp(S[:, i]).
        Y_cols = []
        for i in range(s):
            v = S[:, i]
            Y_cols.append(hvp(v).unsqueeze(1))
        Y = torch.cat(Y_cols, dim=1)  # Y has shape (d, s)

        # Compute an orthonormal basis Q for the range of Y using QR decomposition.
        # Q will have shape (d, s).
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        # First term: compute trace(Q^T A Q)
        term1 = 0.0
        for i in range(s):
            q_i = Q[:, i]
            Aq = hvp(q_i)
            term1 += torch.dot(q_i, Aq)
        
        # Second term: compute (2/(m-2)) * trace(G^T (I - QQ^T) A (I - QQ^T) G)
        term2_sum = 0.0
        for j in range(g_num):
            g_vec = G[:, j]
            # Project g_vec onto the orthogonal complement of the span(Q)
            proj = Q @ (Q.t() @ g_vec)
            r = g_vec - proj  # r = (I - QQ^T) g_vec
            Ar = hvp(r)
            term2_sum += torch.dot(r, Ar)
        
        term2 = (2.0 / (m - 2)) * term2_sum
        
        trace_estimate = term1 + term2
        return trace_estimate

    def check_hutch_pp(self, logger, log_i, train_num, valid_num, m=100):
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            loss = 0
            hutch_pp_list = []

            start_time = time.time()
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'mean')
                trace_estimate = self.hutch_pp_trace_estimator(model, loss, m)
                hutch_pp_list.append(trace_estimate.item() * batch_size)
            trace_estimate = sum(hutch_pp_list) / train_num
            logger.log("trace_estimate", trace_estimate, log_i)
            logger.log("hutch_pp_time", time.time() - start_time, log_i)

            sample_num = 100
            train_hessian_list = []
            train_hessian_2_list = []

            start_time = time.time()
            for i in range(sample_num):
                noise_vector = None
                train_hessian = 0
                train_hessian_2 = 0
                for train_batch in self.dataloader:
                    data, target, batch_size = self.load_batch_func(train_batch, device)
                    output = model(data)
                    loss = loss_fn(output, target)

                    # Compute gradients to get the shape.
                    if noise_vector is None:
                        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                        grad_vector = torch.cat([g.reshape(-1) for g in grads])

                        # Sample the noise vector once.
                        #noise_vector = torch.randn_like(grad_vector)
                        noise_vector = torch.randint_like(grad_vector, high=2)
                        noise_vector[noise_vector == 0] = -1
                    train_quad = self.hessian_quadratic_form(model, loss, noise_vector)

                    train_hessian += train_quad.item()*batch_size

                train_hessian /= train_num

                noise_vector = None
                train_hessian_list.append(train_hessian)


            train_hessian = np.mean(train_hessian_list)

            logger.log("train_hessian", train_hessian, log_i)
            logger.log("hutch_time", time.time() - start_time, log_i)
            plot_curves(logger, ['trace_estimate', 'train_hessian'], path_name='check', file_name='hutch_pp')
            plot_curves(logger, ['hutch_pp_time', 'hutch_time'], path_name='check', file_name='time')

    ##############################################
    # Usage
    ##############################################

    def check_slq(self, logger, i, train_num, valid_num, n_iter=100, n_v=1):
        with sdpa_kernel(SDPBackend.MATH):
            print("=======> SLQ for full model")
            #values_full, weights_full = self.get_full_spectrum(n_iter=n_iter, n_v=n_v, dataloader=self.dataloader)
            values_full, weights_full = self.get_train_spectrum(n_v, n_iter)
            self.values_full = values_full.tolist()
            self.weights_full = weights_full.tolist()
            d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.slq_H_trace = np.sum(values_full * weights_full) * d
            self.slq_H2_trace = np.sum(values_full**2 * weights_full)* d
            self.hvp_H_trace, self.hvp_H2_trace, _, _ = self.compare_hessian(logger, i, train_num, valid_num)
            print(self.slq_H_trace, self.slq_H2_trace, self.hvp_H_trace)
            slq_lambda_max = max(values_full)

            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            
            train_lambda_max = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(loss.mean(), model, power_iter=100)
                train_lambda_max += batch_lambda_max * batch_size
            train_lambda_max /= len(self.dataloader.dataset)

        logger.log("slq_H_trace", self.slq_H_trace, i)
        logger.log("slq_H2_trace", self.slq_H2_trace, i)
        logger.log("hvp_H_trace", self.hvp_H_trace, i)
        logger.log("hvp_H2_trace", self.hvp_H2_trace, i)
        data_names = ['slq_H_trace', 'hvp_H_trace']
        plot_curves(logger, data_names, path_name='check', file_name='hessian')
        data_names = ['slq_H2_trace', 'hvp_H2_trace']
        plot_curves(logger, data_names, path_name='check', file_name='hessian_2')

        logger.log("hvp_lambda_max", train_lambda_max, i)
        logger.log("slq_lambda_max", slq_lambda_max, i)
        plot_curves(logger, ['hvp_lambda_max', 'slq_lambda_max'], path_name='check', file_name='lambda_max')

def add_noise_to_model(model, noise_vector):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        noise = noise_vector[offset: offset + numel].view_as(param)
        param.data.add_(noise)
        offset += numel

def compute_model_norm(model, p=2):
    norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in model.parameters() if p.requires_grad]), 2)
    return norm

def load_batch_func(batch, device='cpu'):
    batch = batch[0].to(device)
    inputs = batch[:, :-1]
    targets = batch
    batch_size = batch.shape[0]
    return inputs, targets, batch_size

def filter_eigenvalues(eigen_list, weight_list, threshold=None):
    filtered_eigen = []
    filtered_weight = []
    #print(np.max(weight_list))
    for eig, w in zip(eigen_list, weight_list):
        if threshold is not None:
            if eig >= threshold and w >= 1e-7:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
        else:
            if w >= 1e-10:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
    #print(filtered_eigen)
    return filtered_eigen, filtered_weight

def renormalize_weights(filtered_weight, epsilon=1e-12):
    total = sum(filtered_weight)
    if total > 0:
        renormalized_weight = [w / (total + epsilon) for w in filtered_weight]
    else:
        # Handle case where all weights are zero
        renormalized_weight = [0.0 for _ in filtered_weight]
    return renormalized_weight

def construct_spectral_density(flat_eigen, flat_weight, lambdas, sigma=0.1):
    density = np.zeros_like(lambdas)
    for eig, w in zip(flat_eigen, flat_weight):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    
    # Normalize the density
    density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
    density /= density_sum + 1e-12  # Avoid division by zero
    return density

def sqrt_with_neg_handling(arr):
    result = np.where(arr < 0, 0, np.sqrt(arr))
    return result


def compute_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        
        eigenvalues = None
        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])

        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)
        
        # Compute the dot product between gradients and noise vector.
        grad_dot_noise = torch.dot(grad_vector, v)
        
        # Compute Hessian-vector product using the Pearlmutter trick.
        Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
            tmp_eigenvalues = torch.sum(Hv*v).cpu().item()

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors

def compute_layer_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    layers = model.get_layers()
    weights = [module.weight for name, module in layers.items()]
    
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        
        eigenvalues = None
        vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
            tmp_eigenvalues = [ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors


def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(p.requires_grad for p in module.parameters(recurse=False)):
            #if (type(module) == torch.nn.Linear) and "LayerNorm" not in name and "ln" not in name and "embeddings" not in name and "pooler" not in name:
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
            #print(f"Layer: {name}, Module: {module}")
                layers[name] = module
    return layers


def weighted_quantile(values, weights, quantile):
    """
    Compute the weighted quantile of a tensor.
    Args:
      values: 1D tensor of eigenvalues.
      weights: 1D tensor of corresponding weights.
      quantile: desired quantile (between 0 and 1).
    Returns:
      The eigenvalue threshold corresponding to the weighted quantile.
    """
    # Sort values and weights in ascending order.
    sorted_vals, sorted_indices = torch.sort(values)
    sorted_weights = weights[sorted_indices]
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = sorted_weights.sum()
    normalized_cum_weights = cumulative_weights / total_weight
    # Find the first index where cumulative weight exceeds the quantile.
    idx = torch.nonzero(normalized_cum_weights >= quantile, as_tuple=False)[0]
    threshold = sorted_vals[idx]
    return threshold

def tail_mass_fraction(values, weights, quantile=0.9):
    """
    Compute the tail mass fraction: the fraction of the total weighted mass
    (∑ p_i λ_i) that comes from eigenvalues above the weighted quantile threshold.
    """
    weights = weights / weights.sum()
    tau = weighted_quantile(values, weights, quantile)
    mask = values >= tau
    numerator = torch.sum(weights[mask] * values[mask])
    denominator = torch.sum(weights * values)
    return (numerator / denominator).item()

def weighted_gini(values, weights):
    """
    Compute the weighted Gini coefficient for the eigenvalue distribution.
    First, normalize weights to obtain a probability distribution:
      q_i = weights_i / (∑_j weights_j).
    Then, compute:
      G = (∑_{i,j} q_i q_j |λ_i - λ_j|) / (2 μ),
    where μ = ∑_i q_i λ_i.
    """
    # Normalize the weights to form a probability distribution.
    q = weights / weights.sum()
    mu = (values * q).sum()
    # Compute pairwise absolute differences between eigenvalues.
    diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
    # Compute pairwise product of normalized weights.
    q_matrix = q.unsqueeze(0) * q.unsqueeze(1)
    gini = torch.sum(diff_matrix * q_matrix) / (2 * mu)
    return gini.item()

def weighted_skewness(values, weights, eps=1e-8):
    """
    Compute the weighted skewness for the eigenvalue distribution.
    Using normalized weights q_i = weights_i / (∑_j weights_j), we have:
      μ   = ∑_i q_i λ_i,
      σ²  = ∑_i q_i (λ_i - μ)²,
      skew = ∑_i q_i (λ_i - μ)³ / (σ³ + eps).
    """
    q = weights / weights.sum()
    mu = (values * q).sum()
    diff = values - mu
    variance = (q * diff**2).sum()
    std = torch.sqrt(variance + eps)
    skew = (q * diff**3).sum() / (std**3 + eps)
    return skew.item()

def compute_sigma_from_weights(state_dict, factor=1.0):
    """
    Compute sigma as a factor times the average standard deviation of the floating-point parameters.
    """
    sigmas = []
    for key, param in state_dict.items():
        if param.requires_grad:
            sigmas.append(param.std().item())
    if sigmas:
        return factor * (sum(sigmas) / len(sigmas))
    else:
        return factor

def compute_kl_divergence_initial_state(final_state_dict, init_state_dict):
    """
    Compute KL(Q||P) where
        Q = N(w_T, sigma^2 I) is the posterior (final weights),
        P = N(w_0, sigma0^2 I) is the prior (initial weights).
    """
    sigma = compute_sigma_from_weights(final_state_dict, factor=0.5)
    sigma0 = compute_sigma_from_weights(init_state_dict, factor=1.0)

    sigma2 = sigma ** 2
    sigma0_2 = sigma0 ** 2
    kl_total = 0.0

    # Loop over parameters (assuming both state_dicts have the same keys)
    for key in final_state_dict:
        param_final = final_state_dict[key]
        param_init = init_state_dict[key].to(param_final.device)
        
        # Consider only floating point parameters (learnable weights)
        if not torch.is_floating_point(param_final) or not torch.is_floating_point(param_init):
            continue
        
        d = param_final.numel()  # number of elements in this tensor
        # Compute squared difference between final and initial weights
        diff_norm_sq = torch.sum((param_final - param_init) ** 2)
        
        # KL divergence for this tensor:
        # KL = 0.5 * [d*log(sigma0^2/sigma^2) + ||w_T - w_0||^2/sigma0^2 + d*(sigma^2/sigma0^2) - d]
        kl_tensor = 0.5 * (d * math.log(sigma0_2 / sigma2) +
                           diff_norm_sq / sigma0_2 +
                           d * (sigma2 / sigma0_2) - d)
        kl_total += kl_tensor

    return kl_total

def pac_bayes_term(kl_div, n, delta):
    """
    Computes the PAC-Bayes bound of the form:
        E[L(f)] <= E[hat{L}(f)] + sqrt((KL(Q||P) + log(1/delta_prime)) / (2n))

    Args:
        empirical_loss (float or torch.Tensor): Empirical loss (averaged over n samples).
        kl_div (float or torch.Tensor): KL(Q||P) already computed.
        n (int): Number of samples in the dataset.
        delta_prime (float): Confidence parameter (e.g., 0.05).

    Returns:
        torch.Tensor: The PAC-Bayes upper bound on the true (expected) loss.
    """
    # Ensure all inputs are torch.Tensor for consistency
    if not isinstance(kl_div, torch.Tensor):
        kl_div = torch.tensor(float(kl_div), dtype=torch.float32)
        
    # Convert n and delta_prime to Tensors if needed
    n_t = torch.tensor(float(n), dtype=torch.float32)
    delta_t = torch.tensor(float(delta), dtype=torch.float32)
    
    # Compute the PAC-Bayes complexity term
    # sqrt( (KL + log(1/delta)) / (2n) )
    complexity_term = torch.sqrt((kl_div + torch.log(1.0 / delta_t)) / (2.0 * n_t))

    # Final bound is empirical_loss + complexity
    return complexity_term

def plot_curves(log, data_names, path_name, file_name=None, yabel='Hessian', save_dir="./results/", x_log=True, y_log=True):
    if file_name is None:
        file_name = path_name
    try:
        train_converge = log["train_converge"]["value"]
        val_converge = log["val_converge"]["value"]
    except:
        train_converge = 0
        val_converge = 0
    #print(train_converge, val_converge)
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    #print("Plotting hessian, ", log.label)
    for i, name in enumerate(data_names):
        plt.plot(log[name]["iter"], log[name]["value"], label=name)

    if train_converge > 0:
        plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
    if val_converge > 0:
        plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Hessian")
    if x_log:
        plt.xscale("log", base=10)
    if y_log:
        plt.yscale("log", base=10)
    #plt.ylim(1e-7, 1e7)
    plt.grid()
    plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
    plt.savefig(f"{save_dir}{path_name}/{file_name}_{log.label}.png", dpi=150)
    plt.draw()
    plt.close()

def sqrt_sum_nonnegative(arr):
    arr = np.array(arr)
    arr = np.where(arr < 0, 0, arr)  
    return np.sum(np.sqrt(arr))  

def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if (type(module) == torch.nn.Linear) and \
        ("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
            layers[name] = module
    return layers

def compute_hessians_quantity(model, loss, device="cpu", state_dict = None):
    # Get parameters and gradients of corresponding layer
    with sdpa_kernel(SDPBackend.MATH):
        layers = model.get_layers()
        #layers = get_layers(model)
        weights = [module.weight for name, module in layers.items()]
        #weights = list(model.parameters())
        model.zero_grad()
        gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True, allow_unused=True)
        vs = []
        for name, module in layers.items():
            weight = module.weight
            v = weight.detach().clone() - model.init_state[name+".weight"].to(weight.device)
            vs.append(v)

        model.zero_grad()    
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

        layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
        
        out = np.array(layer_hessian_quantities)
        value = sqrt_sum_nonnegative(out)
    return value