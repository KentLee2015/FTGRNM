import cupy as cp
import numpy as np
import time

cp.cuda.set_allocator(None)
cp.cuda.runtime.deviceSynchronize()
dtype = cp.float32

# =====================
# 
# =====================
def clear_gpu_memory():
    """（CuPy）"""
    try:
        # CUDA
        cp.cuda.Device(0).synchronize()
        
        # 
        pool = cp.get_default_memory_pool()
        if pool is not None:
            pool.free_all_blocks()
        
        pinned_pool = cp.get_default_pinned_memory_pool()
        if pinned_pool is not None:
            pinned_pool.free_all_blocks()
        
        # Python
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"Memory cleanup warning: {e}")

def compute_mape(var, var_hat):
    return cp.sum(cp.abs(var - var_hat) / (cp.abs(var) + 1e-8)) / var.shape[0]

def compute_smape(var, var_hat):
    denominator = cp.abs(var) + cp.abs(var_hat) + 1e-8
    return cp.sum(2.0 * cp.abs(var - var_hat) / denominator) / var.shape[0]

def compute_mae(var, var_hat):
    return cp.sum(cp.abs(var - var_hat)) / var.shape[0]

def compute_rmse(var, var_hat):
    return cp.sqrt(cp.sum((var - var_hat) ** 2) / var.shape[0])

def compute_all_metrics(var, var_hat):
    return {
        'mape': float(compute_mape(var, var_hat)),
        'smape': float(compute_smape(var, var_hat)),
        'mae': float(compute_mae(var, var_hat)),
        'rmse': float(compute_rmse(var, var_hat))
    }

def laplacian(n, tau, lookback, period):
    ell = cp.zeros(n, dtype=dtype)
    ell[0] = 2 * tau + (lookback - 1)
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
        for m in range(lookback - 1):
            ell[(m + 1) * period] = -1
    return ell

def soft_threshold(X, thresh):
    return cp.sign(X) * cp.maximum(cp.abs(X) - thresh, 0.0)

# =====================
# FTGRNM 
# =====================
def prox_fourier(z, s, w, lmbda, denominator, eta, x_prev):
    """"""
    T = z.shape[0]
    temp1 = cp.fft.rfft(lmbda * (z - s) - w + eta * x_prev)
    denom = denominator + eta
    temp1 = temp1 / denom
    abs_temp1 = cp.abs(temp1)
    temp2 = 1 - T / (denom * abs_temp1 + 1e-8)
    temp2 = cp.maximum(temp2, 0)
    return cp.fft.irfft(temp1 * temp2, n=T)

def prox_graph_only(z, s, w, lmbda, denominator):
    """"""
    temp1 = cp.fft.rfft(lmbda * (z - s) - w)
    return cp.fft.irfft(temp1 / denominator, n=z.shape[0])

def update_z_soft(y_obs, x, s, w, lmbda, rho):
    """z"""
    mask = (y_obs != 0).astype(dtype)
    numerator = rho * mask * y_obs + lmbda * (x + s + w / lmbda)
    denominator = rho * mask + lmbda
    return numerator / (denominator + 1e-12)

def update_z_hard(y_obs, x, s, w, lmbda):
    """z"""
    z = x + s + w / lmbda
    mask = (y_obs != 0)
    z = cp.where(mask, y_obs, z)
    return z

def update_s_proximal(z, x, w, lmbda, mu, eta_s, s_prev):
    """s - FTGRNM"""
    residual = z - x - w / lmbda
    numerator = lmbda * residual + eta_s * s_prev
    denominator = lmbda + eta_s
    thresh = mu / denominator
    return soft_threshold(numerator / denominator, thresh)

def update_s_basic(z, x, w, lmbda, mu):
    """s"""
    residual = z - x - w / lmbda
    return soft_threshold(residual, mu / lmbda)

def update_s_zero(z, x, w, lmbda):
    """s=0"""
    return cp.zeros_like(z)

def update_w(x, z, s, w, lmbda):
    """ """
    return w + lmbda * (x + s - z)

# =====================
# 
# =====================
def get_validation_indices(y_obs, val_ratio=0.1):
    """
    
    
    :
        y_obs: 0
        val_ratio: 
    :
        train_indices, val_indices
    """
    pos_obs = cp.where(y_obs != 0)[0]
    n_obs = len(pos_obs)
    n_val = int(n_obs * val_ratio)
    
    # 
    # : cp.random GPU
    perm = cp.random.permutation(n_obs)
    val_indices = pos_obs[perm[:n_val]]
    train_indices = pos_obs[perm[n_val:]]
    
    return train_indices, val_indices

# =====================
# 
# =====================
def check_admm_convergence(x, z, s, w, w_prev, lmbda, it,
                           eps_abs=1e-4, eps_rel=1e-2):
    """
    ADMM
    """
    # 
    r = x + s - z
    r_norm = float(cp.linalg.norm(r))
    
    # 
    s_norm = float(cp.linalg.norm(w - w_prev))
    
    # 
    p = x.size
    eps_pri = cp.sqrt(p) * eps_abs + eps_rel * max(
        float(cp.linalg.norm(x)),
        float(cp.linalg.norm(s)),
        float(cp.linalg.norm(z))
    )
    eps_dual = cp.sqrt(p) * eps_abs + eps_rel * float(cp.linalg.norm(w))
    
    pri_ok = r_norm <= eps_pri
    dual_ok = s_norm <= eps_dual
    
    # 5
    if it < 5:
        return False, r_norm, s_norm, eps_pri, eps_dual
    
    return pri_ok and dual_ok, r_norm, s_norm, eps_pri, eps_dual

def check_early_stopping(history_mape, patience=10, tol=1e-3):
    """
    
    """
    if len(history_mape) < patience + 1:
        return False
    
    recent = history_mape[-(patience + 1):]
    best = min(recent[:-1])
    current = recent[-1]
    
    if current > best * (1 + tol):
        return True
    
    return False

# =====================
# FTGRNM - ADMM
# =====================
def FTGRNM_Proximal_ADMM(y_true, y_obs, lmbda, gamma, mu, rho,
                         tau_t, lookback, period,
                         eta_x=1e-4, eta_s=1e-4,
                         maxiter=200, 
                         eps_abs=1e-4, eps_rel=1e-2,
                         patience=15,
                         train_indices=None,  # 
                         val_indices=None,    # 
                         verbose=True):
    """
    FTGRNMADMM
    
    :
        train_indices: 
        val_indices: 
    """
    T = len(y_obs)
    
    y_true = cp.asarray(y_true, dtype=dtype)
    y_obs = cp.asarray(y_obs, dtype=dtype)
    
    # =====  =====
    if train_indices is not None and val_indices is not None:
        # 
        pass
    else:
        # 
        train_indices, val_indices = get_validation_indices(y_obs, val_ratio=0.1)
    
    # 
    y_val = y_true[val_indices]
    
    # 
    pos_test = cp.where((y_true != 0) & (y_obs == 0))
    y_test = y_true[pos_test]
    
    # =====  =====
    obs_mean = cp.mean(y_obs[y_obs != 0])
    x = cp.where(y_obs != 0, y_obs, obs_mean)
    z = y_obs.copy()
    s = cp.zeros_like(y_obs)
    w = cp.zeros_like(y_obs)
    
    # =====  =====
    ell_t = laplacian(T, tau_t, lookback, period)
    ell = cp.fft.rfft(ell_t)
    denominator = lmbda + gamma * cp.abs(ell) ** 2
    
    # =====  =====
    history = {
        'r_norm': [],
        's_norm': [],
        'mape_train': [],
        'mape_val': [],
        'mae_val': [],
        'smape_val': [],
        'rmse_val': []
    }
    
    start_time = time.time()
    converged = False
    early_stop = False
    
    for it in range(maxiter):
        x_prev = x.copy()
        s_prev = s.copy()
        w_prev = w.copy()
        
        # ===== 1. x =====
        x = prox_fourier(z, s, w, lmbda, denominator, eta_x, x_prev)
        
        # ===== 2. s =====
        s = update_s_proximal(z, x, w, lmbda, mu, eta_s, s_prev)
        
        # ===== 3. z =====
        z = update_z_soft(y_obs, x, s, w, lmbda, rho)
        
        # ===== 4. w =====
        w = update_w(x, z, s, w, lmbda)
        
        # ===== 5. 5 =====
        if it % 5 == 0:
            # 
            mape_train = compute_mape(y_obs[train_indices], x[train_indices])
            
            # 
            metrics_val = compute_all_metrics(y_val, x[val_indices])
            
            history['mape_train'].append(float(mape_train))
            history['mape_val'].append(metrics_val['mape'])
            history['mae_val'].append(metrics_val['mae'])
            history['smape_val'].append(metrics_val['smape'])
            history['rmse_val'].append(metrics_val['rmse'])
            
            # =====  =====
            converged, r_norm, s_norm, eps_pri, eps_dual = check_admm_convergence(
                x, z, s, w, w_prev, lmbda, it, eps_abs, eps_rel
            )
            history['r_norm'].append(r_norm)
            history['s_norm'].append(s_norm)
            
            # =====  =====
            if len(history['mape_val']) > patience:
                early_stop = check_early_stopping(
                    history['mape_val'], 
                    patience=patience, 
                    tol=1e-3
                )
            
            # =====  =====
            if verbose and (it % 50 == 0 or it == 0):
                metrics_test = compute_all_metrics(y_test, x[pos_test])
                print(f"Iter {it+1:4d}: "
                      f"MAPE={metrics_test['mape']:.5f}, "
                      f"SMAPE={metrics_test['smape']:.5f}, "
                      f"MAE={metrics_test['mae']:.3f}, "
                      f"RMSE={metrics_test['rmse']:.3f}, "
                      f"r={r_norm:.3e}, s={s_norm:.3e}")
            
            # =====  =====
            if converged or early_stop:
                if early_stop and verbose:
                    print(f"\n Early stopping at iteration {it+1} (validation MAPE not improving)")
                elif converged and verbose:
                    print(f"\n Converged at iteration {it+1}")
                    print(f"   r_norm={r_norm:.3e} <= eps_pri={eps_pri:.3e}")
                    print(f"   s_norm={s_norm:.3e} <= eps_dual={eps_dual:.3e}")
                break
    
    elapsed = time.time() - start_time
    
    if verbose and not converged and not early_stop:
        print(f"\n  Max iterations ({maxiter}) reached")
    
    if verbose:
        final_metrics = compute_all_metrics(y_test, x[pos_test])
        print(f"\n{'='*60}")
        print(f"Total time: {elapsed:.2f}s, Iterations: {it+1}")
        print(f"FINAL RESULTS:")
        print(f"  MAPE : {final_metrics['mape']:.5f}")
        print(f"  SMAPE: {final_metrics['smape']:.5f}")
        print(f"  MAE  : {final_metrics['mae']:.3f}")
        print(f"  RMSE : {final_metrics['rmse']:.3f}")
        print(f"{'='*60}")
    
    return x, history


# =====================
# 
# =====================
if __name__ == "__main__":
    np.random.seed(1000)
    cp.random.seed(1000)  # GPU
    
    # for rate in [0.3,0.5,0.7,0.9]:
    for rate in [0.3]:
        MAPE = []
        SMAPE = []
        MAE = []
        RMSE = []
        
        for seed in [1000,700,300]:
            print(f'\n{"="*60}')
            print(f"Seed: {seed}, Missing rate: {rate}")
            
            # 
            np.random.seed(seed)
            cp.random.seed(seed)
            
            dense_mat = np.load('../input/datasets/kent2020/transdim/California-data-set/pems-w1.npz')['arr_0']
            for t in range(2, 5):
                dense_mat = np.append(
                    dense_mat,
                    np.load(f'../input/datasets/kent2020/transdim/California-data-set/pems-w{t}.npz')['arr_0'],
                    axis=1)
            dense_mat = dense_mat[:,4032:]
            #dense_mat = dense_mat[:,:4032]
            print(dense_mat.shape)
            # missing_pattern ='random'
            # noise_pattern='full_mixed'
            


            if rate == 0.3: 
                #for missing_pattern in ['random','row_block','spatial_cluster','column_block']:
                for missing_pattern in ['random']:    
                    for noise_pattern in ['None']:
                    #for noise_pattern in ['full_mixed']:
                        #for C in [1e-4,1e-5,1e-6,1e-7]:
                        if missing_pattern =='random':                          
                            if noise_pattern == 'full_mixed':
                                break
                            else:
                                C = 1e-4
                                alpha = 100
                                r = 1
                        elif missing_pattern =='row_block':                            
                            C = 1e-5 
                            alpha = 100
                            if noise_pattern == 'full_mixed':
                                r = 4
                            else:
                                r = 3
                        elif missing_pattern =='spatial_cluster':
                            C = 1e-6
                            alpha = 100
                            if noise_pattern == 'full_mixed':
                                r = 2
                            else:
                                r = 1
                        elif missing_pattern =='column_block':
                            C = 1e-7
                            alpha =1000
                            r = 4

                        # for alpha in [10,100,1000]:
                        #     mu = alpha * varphi
                        #     rho = alpha * varphi
                        #     for r in [1,2,3,4]:
                            #r = 4

            
            # if rate == 0.5: 
            #     for missing_pattern in ['random']:
            #         for noise_pattern in ['None']:
            #             #for C in [1e-4,1e-5,1e-6,1e-7]:                
            #             C = 1e-4
            #             varphi = C * T
            #             theta = varphi
            #             alpha = 100
            #             mu = alpha * varphi
            #             rho = alpha * varphi                        
            #             r = 1
            #             # for alpha in [10,100,1000]:
            #             #     mu = alpha * varphi
            #             #     rho = alpha * varphi
            #             #     for r in [1,2,3,4]:
           

                            
            # if rate == 0.7: 
            #     for missing_pattern in ['random']:
            #         for noise_pattern in ['None']:
            #             #for C in [1e-4,1e-5,1e-6,1e-7]:                
            #             C = 1e-4
            #             varphi = C * T
            #             theta = varphi
            #             alpha = 100
            #             mu = alpha * varphi
            #             rho = alpha * varphi                        
            #             r = 2
            #             # for alpha in [10,100,1000]:
            #             #     mu = alpha * varphi
            #             #     rho = alpha * varphi
            #             #     for r in [1,2,3,4]:
                            
            # if rate == 0.9: 
            #     for missing_pattern in ['random']:
            #         # for noise_pattern in ['None','full_mixed']:
            #         for noise_pattern in ['full_mixed']:
            #             #for C in [1e-4,1e-5,1e-6,1e-7]:
            #             if noise_pattern == 'None':
            #                 C = 1e-4
            #                 r = 2
            #             else:
            #                 C = 1e-5
            #                 r = 4
            #             varphi = C * T
            #             theta = varphi
            #             alpha = 100
            #             mu = alpha * varphi
            #             rho = alpha * varphi                        
            #             # for alpha in [10,100,1000]:
            #             #     mu = alpha * varphi
            #             #     rho = alpha * varphi
            #             #     for r in [1,2,3,4]:
            #             #     #r = 4

            # if rate == 0.9: 
            #     for missing_pattern in ['column_block']:
            #         # for noise_pattern in ['None','full_mixed']:
            #         for noise_pattern in ['None']:
            #             #for C in [1e-4,1e-5,1e-6,1e-7]:
            #             C = 1e-7
            #             #for r in [1,2,3,4]:
            #             r = 2
            #             varphi = C * T
            #             theta = varphi
            #             #for alpha in [10,100,1000]:
            #             alpha = 1000
            #             mu = alpha * varphi
            #             rho = alpha * varphi                        
                   
                
                


                        
                        # =====  =====
                        period = 288
                        lookback = 1
                                                
                        eta_x = 1e-4
                        eta_s = 1e-4
                        eps_abs = 1e-4
                        eps_rel = 1e-2
                        patience = 15
                        maxiter = 200
            
                    #                             
                        obs_mask, sparse_mat = create_missing_data_mask(
                            dense_mat, missing_rate=rate, noise_type=noise_pattern, missing_type=missing_pattern
                        )
                        
                        # GPU
                        dense_mat_gpu = cp.asarray(dense_mat, dtype=dtype)
                        sparse_mat_gpu = cp.asarray(sparse_mat, dtype=dtype)
                        
                        # Flatten
                        dense_vec = dense_mat_gpu.reshape(-1)
                        sparse_vec = sparse_mat_gpu.reshape(-1)
                        T = dense_vec.shape[0]

                        varphi = C * T
                        theta = varphi
                        mu = alpha * varphi
                        rho = alpha * varphi                        
                        print('missing_pattern:',missing_pattern)
                        print('noise_pattern:',noise_pattern)
                        print('C:',C)
                        print('r:',r)
                        print('alpha:',alpha)
                        
                        # 
                        train_indices, val_indices = get_validation_indices(sparse_vec, val_ratio=0.1)
                                        
                        print(f"\nParameters:")
                        print(f"  varphi={varphi:.2e}, mu={mu:.2e}, rho={rho:.2e}, r={r}")
                        print(f"  eta_x={eta_x:.2f}, eta_s={eta_s:.2f}")
                        print(f"  eps_abs={eps_abs:.0e}, eps_rel={eps_rel:.0e}")
                        print(f"  patience={patience}, maxiter={maxiter}")
                        print(f"  Training samples: {len(train_indices)}")
                        print(f"  Validation samples: {len(val_indices)}")
                        
                        # ===== ADMM =====
                        start = time.time()
                        x, history = FTGRNM_Proximal_ADMM(
                            dense_vec, sparse_vec,
                            varphi, theta, mu, rho,
                            r, lookback, period,
                            eta_x=eta_x,
                            eta_s=eta_s,
                            maxiter=maxiter,
                            eps_abs=eps_abs,
                            eps_rel=eps_rel,
                            patience=patience,
                            train_indices=train_indices,  # 
                            val_indices=val_indices,      # 
                            verbose=True
                        )
                        print(f"Time: {time.time() - start:.2f}s")
                        
                        # 
                        pos_test = cp.where((dense_vec != 0) & (sparse_vec == 0))
                        y_test = dense_vec[pos_test]
                        metrics = compute_all_metrics(y_test, x[pos_test])
                        
                        MAPE.append(metrics['mape'])
                        SMAPE.append(metrics['smape'])
                        MAE.append(metrics['mae'])
                        RMSE.append(metrics['rmse'])
                                                                   
                    
                        # GPU
            try:
                cp.cuda.Device(0).synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"Cleanup warning: {e}")
            
            cp.cuda.Stream.null.synchronize()
            clear_gpu_memory()   

        # 
        print(f'\n{"="*60}')
        print(f"SUMMARY for missing rate {rate}:")
        print(f"  MAPE : {np.mean(MAPE):.6f} ± {np.std(MAPE):.6f}")
        print(f"  SMAPE: {np.mean(SMAPE):.6f} ± {np.std(SMAPE):.6f}")
        print(f"  MAE  : {np.mean(MAE):.4f} ± {np.std(MAE):.4f}")
        print(f"  RMSE : {np.mean(RMSE):.4f} ± {np.std(RMSE):.4f}")
        print(f"{'='*60}")
