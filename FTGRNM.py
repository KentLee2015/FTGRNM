import cupy as cp
import numpy as np
import time
from missing_patterns import create_missing_data_mask

cp.cuda.set_allocator(None)
cp.cuda.runtime.deviceSynchronize()
dtype = cp.float32


# =====================
# Tools
# =====================
def clear_gpu_memory():
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass


def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / (var + 1e-8)) / var.shape[0]


def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


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


def prox_1d(z, w, lmbda, denominator):
    T = z.shape[0]

    temp1 = cp.fft.rfft(lmbda * z - w)
    denom = denominator[:temp1.shape[0]]

    temp1 = temp1 / denom

    abs_temp1 = cp.abs(temp1)
    temp2 = 1 - T / (denom * abs_temp1 + 1e-8)
    temp2 = cp.maximum(temp2, 0)

    return cp.fft.irfft(temp1 * temp2, n=T)


def update_z(y_obs, x, s, w, lmbda, eta):
    z = x + s + w / lmbda
    mask = (y_obs != 0)

    z = cp.where(
        mask,
        ((lmbda * (x + s) + w) + eta * y_obs) / (lmbda + eta),
        z
    )
    return z


def update_w(x, z, s, w, lmbda):
    return w + lmbda * (x + s - z)


# =====================
# Main Algorithm
# =====================
def FTGRNM(y_true, y_obs, lmbda, gamma, mu, eta,
           tau_t, lookback, period, maxiter=50):
    T = len(y_obs)

    pos_test = cp.where((y_true != 0) & (y_obs == 0))
    y_test = y_true[pos_test]

    z = y_obs.copy()
    w = cp.zeros_like(y_obs)
    s = cp.zeros_like(y_obs)
    x = cp.zeros_like(y_obs)

    ell_t = laplacian(T, tau_t, lookback, period)
    ell = cp.fft.rfft(ell_t)
    denominator = lmbda + gamma * cp.abs(ell) ** 2

    for it in range(maxiter):

        x = prox_1d(z - s, w, lmbda, denominator)

        residual = z - x - w / lmbda
        s = soft_threshold(residual, mu / lmbda)

        z = update_z(y_obs, x, s, w, lmbda, eta)

        w = update_w(x, z, s, w, lmbda)

        # if (it + 1) % 100 == 0 or it == 0:
        if (it + 1) % 50 == 0:
            mape = compute_mape(
                cp.asnumpy(y_test),
                cp.asnumpy(x[pos_test])
            )
            rmse = compute_rmse(
                cp.asnumpy(y_test),
                cp.asnumpy(x[pos_test])
            )
            print(f"Iter {it + 1}, MAPE={mape:.5f},RMSE={rmse:.5f}")

    return x, mape, rmse


# =====================
# Load Data
# =====================
Tag = 0
for rate in [0.3, 0.5, 0.7, 0.9, 0.9]:
    MAPE = []
    RMSE = []

    for seed in [300, 700, 1000]:
        # for seed in [1000]:
        print('seed=', seed)
        np.random.seed(seed)
        # cp.random.seed(seed)
        dense_mat = np.load('../datasets/kent2020/transdim/California-data-set/pems-w1.npz')['arr_0']
        for t in range(2, 5):
            dense_mat = np.append(
                dense_mat,
                np.load(f'../datasets/kent2020/transdim/California-data-set/pems-w{t}.npz')['arr_0'],
                axis=1)
        # dense_mat = dense_mat[:,:4032]
        dense_mat = dense_mat[:, 4032:]

        if rate == 0.9:
            if Tag > 2:
                obs_mask, sparse_mat = create_missing_data_mask(
                    dense_mat, missing_rate=rate, noise_type='full_mixed', missing_type='random'
                )
                Tag += 1
            else:
                obs_mask, sparse_mat = create_missing_data_mask(
                    dense_mat, missing_rate=rate, noise_type='None', missing_type='random'
                )
                Tag += 1
        else:
            obs_mask, sparse_mat = create_missing_data_mask(
                dense_mat, missing_rate=rate, noise_type='None', missing_type='random'
            )
        # ===== GPU =====
        dense_mat = cp.asarray(dense_mat, dtype=dtype)
        sparse_mat = cp.asarray(sparse_mat, dtype=dtype)

        # ===== flatten =====
        dense_vec = dense_mat.reshape(-1)
        sparse_vec = sparse_mat.reshape(-1)

        T = dense_vec.shape[0]

        # ===== 参数 =====
        period = 288
        lookback = 1
        print(Tag)
        if Tag < 4:
            C = 1e-4
        else:
            C = 1e-5
        # for C in [1e-4,1e-5,1e-6,1e-7]:
        # for C in [1e-5]:
        print('C=', C)
        varphi = C * T
        # for alpha in [1e1,1e2,1e3]:
        for alpha in [1e2]:
            print('alpha=', alpha)
            theta = 1 * varphi
            mu = alpha * varphi
            rho = alpha * varphi
            if rate == 0.3:
                r = 1
            elif rate == 0.5:
                r = 1
            elif rate == 0.7:
                r = 2
            elif rate == 0.9:
                r = 4
            # for r in [4]:
            # for r in [1,2,3,4]:
            print('r=', r)

            # ===== 运行 =====
            start = time.time()

            x, mape, rmse = FTGRNM(
                dense_vec, sparse_vec,
                varphi, theta, mu, rho,
                r, lookback, period,
                maxiter=50
            )
            print("Time:", time.time() - start)
            MAPE.append(mape)
            RMSE.append(rmse)

            cp.cuda.Stream.null.synchronize()
            clear_gpu_memory()
    print('Average MAPE and std', np.mean(MAPE), np.std(MAPE))
    print('Average RMSE and std', np.mean(RMSE), np.std(RMSE))

for rate in [0.3]:
    for noise in ['None', 'full_mixed']:
        for missing in ['row_block', 'spatial_cluster', 'column_block']:
            print(noise)
            print(missing)
            MAPE = []
            RMSE = []

            for seed in [300, 700, 1000]:
                # for seed in [1000]:
                print('seed=', seed)
                np.random.seed(seed)
                dense_mat = np.load('../input/datasets/kent2020/transdim/California-data-set/pems-w1.npz')['arr_0']
                for t in range(2, 5):
                    dense_mat = np.append(
                        dense_mat,
                        np.load(f'../input/datasets/kent2020/transdim/California-data-set/pems-w{t}.npz')['arr_0'],
                        axis=1)
                # dense_mat = dense_mat[:,:4032]
                dense_mat = dense_mat[:, 4032:]
                obs_mask, sparse_mat = create_missing_data_mask(
                    dense_mat, missing_rate=rate, noise_type=noise, missing_type=missing
                )

                # ===== GPU =====
                dense_mat = cp.asarray(dense_mat, dtype=dtype)
                sparse_mat = cp.asarray(sparse_mat, dtype=dtype)

                # ===== flatten =====
                dense_vec = dense_mat.reshape(-1)
                sparse_vec = sparse_mat.reshape(-1)

                T = dense_vec.shape[0]

                # ===== parameters =====
                period = 288
                lookback = 1
                # for C in [1e-4,1e-5,1e-6,1e-7]:
                # for C in [1e-7]:
                if missing == 'row_block':
                    C = 1e-5
                    alpha = 100
                    if noise == 'None':
                        r = 3
                    else:
                        r = 4
                elif missing == 'spatial_cluster':
                    C = 1e-6
                    alpha = 100
                    if noise == 'None':
                        r = 1
                    else:
                        r = 3
                elif missing == 'column_block':
                    C = 1e-7
                    alpha = 1000
                    r = 1
                print('C=', C)
                varphi = C * T
                # for alpha in [1e1,1e2,1e3]:
                # for alpha in [1e2]:
                print('alpha=', alpha)
                theta = 1 * varphi
                mu = alpha * varphi
                rho = alpha * varphi
                # for r in [4]:
                # for r in [1,2,3,4]:
                print('r=', r)

                # ===== run=====
                start = time.time()

                x, mape, rmse = FTGRNM(
                    dense_vec, sparse_vec,
                    varphi, theta, mu, rho,
                    r, lookback, period,
                    maxiter=50
                )
                print("Time:", time.time() - start)
                MAPE.append(mape)
                RMSE.append(rmse)

                cp.cuda.Stream.null.synchronize()
                clear_gpu_memory()
            print('Average MAPE and std', np.mean(MAPE), np.std(MAPE))
            print('Average RMSE and std', np.mean(RMSE), np.std(RMSE))


