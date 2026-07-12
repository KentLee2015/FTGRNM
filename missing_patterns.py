import numpy as np


def create_missing_data_mask(y_true, missing_rate=0.1, missing_type='random',
                             block_length=288, noise_type=None, noise_level=0.8,
                             cluster_size=3):
    N, T = y_true.shape
    mask = np.ones((N, T), dtype=bool)

    if missing_type == 'random':
        missing_mask = np.random.rand(N, T) < missing_rate
        mask[missing_mask] = False

    elif missing_type == 'row_block':
        mask = create_row_block_missing(N, T, missing_rate, block_length)

    elif missing_type == 'column_block':
        mask = create_column_block_missing(N, T, missing_rate, block_length)

    elif missing_type == 'spatial_cluster':
        mask = create_spatial_cluster_missing(N, T, missing_rate, block_length, cluster_size)

    actual_missing_rate = 1 - np.mean(mask)
    print(f"missing rate: {missing_rate:.2f}, actual missing rate: {actual_missing_rate:.2f}")

    # mask
    y_obs_with_missing = y_true.copy()
    y_obs_with_missing[~mask] = 0

    # add noise
    if noise_type is not None:
        if noise_type == 'half_mixed':
            y_obs_with_missing = add_half_mixed_noise(y_obs_with_missing, mask, noise_level)
        elif noise_type == 'full_mixed':
            y_obs_with_missing = add_full_mixed_noise(y_obs_with_missing, mask, noise_level)

    return mask, y_obs_with_missing


def create_row_block_missing(N, T, missing_rate, block_length):
    mask = np.ones((N, T), dtype=bool)
    total_elements = N * T
    target_missing = int(total_elements * missing_rate)

    if target_missing == 0:
        return mask

    current_missing = 0

    for i in range(N):
        if current_missing >= target_missing:
            break

        row_target = min(T - 1, int(T * missing_rate) + np.random.randint(-2, 3))
        row_target = max(0, min(T, row_target))

        row_missing = 0
        attempts = 0
        max_attempts = 100

        while row_missing < row_target and attempts < max_attempts and current_missing < target_missing:
            start_t = np.random.randint(0, T - block_length + 1)
            end_t = start_t + block_length

            # check
            if np.all(mask[i, start_t:end_t]):
                if row_missing + block_length <= row_target:
                    mask[i, start_t:end_t] = False
                    row_missing += block_length
                    current_missing += block_length
                else:
                    remaining = row_target - row_missing
                    if remaining > 0:
                        available = np.where(mask[i, start_t:end_t])[0]
                        if len(available) >= remaining:
                            selected = np.random.choice(available, remaining, replace=False)
                            mask[i, start_t + selected] = False
                            row_missing += remaining
                            current_missing += remaining
            attempts += 1

    if current_missing < target_missing:
        remaining = target_missing - current_missing
        positions = np.where(mask)
        if len(positions[0]) > 0:
            n_to_miss = min(remaining, len(positions[0]))
            indices = np.random.choice(len(positions[0]), n_to_miss, replace=False)
            for idx in indices:
                mask[positions[0][idx], positions[1][idx]] = False

    return mask


def create_column_block_missing(N, T, missing_rate, block_length):
    mask = np.ones((N, T), dtype=bool)
    total_elements = N * T
    target_missing = int(total_elements * missing_rate)

    if target_missing == 0:
        return mask

    cols_per_block = block_length
    elements_per_block = N * cols_per_block
    num_full_blocks = min(target_missing // elements_per_block, T // block_length)
    remaining_elements = target_missing - num_full_blocks * elements_per_block

    available_cols = list(range(T - block_length + 1))
    selected_starts = []

    for _ in range(num_full_blocks):
        if not available_cols:
            break
        idx = np.random.randint(0, len(available_cols))
        start = available_cols.pop(idx)
        selected_starts.append(start)

        to_remove = []
        for s in available_cols:
            if abs(s - start) < block_length:
                to_remove.append(s)
        for s in to_remove:
            if s in available_cols:
                available_cols.remove(s)

    # mask
    for start in selected_starts:
        end = min(start + block_length, T)
        mask[:, start:end] = False

    if remaining_elements > 0 and available_cols:
        remaining_cols = min(remaining_elements // N + 1, T)
        if remaining_cols > 0 and available_cols:
            idx = np.random.randint(0, len(available_cols))
            start = available_cols[idx]
            end = min(start + remaining_cols, T)
            mask[:, start:end] = False

    return mask


def create_spatial_cluster_missing(N, T, missing_rate, block_length, cluster_size):
    mask = np.ones((N, T), dtype=bool)
    total_elements = N * T
    target_missing = int(total_elements * missing_rate)

    if target_missing == 0 or cluster_size > N:
        return mask

    rows_per_block = min(cluster_size, N)
    cols_per_block = min(block_length, T)
    elements_per_block = rows_per_block * cols_per_block

    num_full_blocks = min(target_missing // elements_per_block,
                          (N // rows_per_block) * (T // cols_per_block))
    remaining_elements = target_missing - num_full_blocks * elements_per_block

    row_blocks = N // rows_per_block
    col_blocks = T // cols_per_block

    all_positions = []
    for rb in range(row_blocks):
        for cb in range(col_blocks):
            start_row = rb * rows_per_block
            start_col = cb * cols_per_block
            end_row = min(start_row + rows_per_block, N)
            end_col = min(start_col + cols_per_block, T)

            if np.all(mask[start_row:end_row, start_col:end_col]):
                all_positions.append((start_row, start_col))

    if num_full_blocks > 0 and all_positions:
        selected_indices = np.random.choice(len(all_positions),
                                            min(num_full_blocks, len(all_positions)),
                                            replace=False)

        for idx in selected_indices:
            start_row, start_col = all_positions[idx]
            end_row = min(start_row + rows_per_block, N)
            end_col = min(start_col + cols_per_block, T)
            mask[start_row:end_row, start_col:end_col] = False

    current_missing = total_elements - np.sum(mask)
    if current_missing < target_missing:
        remaining = target_missing - current_missing
        positions = np.where(mask)
        if len(positions[0]) > 0:
            n_to_miss = min(remaining, len(positions[0]))
            indices = np.random.choice(len(positions[0]), n_to_miss, replace=False)
            for idx in indices:
                mask[positions[0][idx], positions[1][idx]] = False

    return mask


def add_half_mixed_noise(y_true, mask, noise_level):
    y_noisy = y_true.copy()
    obs_positions = np.where(mask)
    n_obs = len(obs_positions[0])

    if n_obs > 0:
        n_noisy = int(n_obs * 0.5)
        if n_noisy > 0:
            noisy_indices = np.random.choice(n_obs, n_noisy, replace=False)
            noisy_positions = (obs_positions[0][noisy_indices], obs_positions[1][noisy_indices])

            # gaussian
            data_std = np.std(y_true[noisy_positions])
            gaussian_noise = np.random.normal(0, data_std * noise_level * 0.7,
                                              size=len(noisy_positions[0]))
            y_noisy[noisy_positions] += gaussian_noise

            # outliers
            n_outliers = int(n_noisy * noise_level * 0.3)
            if n_outliers > 0:
                outlier_indices = np.random.choice(n_noisy, n_outliers, replace=False)
                data_mean = np.mean(y_true[noisy_positions])

                for idx in outlier_indices:
                    i, j = noisy_positions[0][idx], noisy_positions[1][idx]
                    outlier_magnitude = np.random.uniform(4, 6) * data_std
                    sign = 1 if np.random.rand() > 0.5 else -1
                    y_noisy[i, j] = data_mean + sign * outlier_magnitude

    return y_noisy


def add_full_mixed_noise(y_true, mask, noise_level):
    y_noisy = y_true.copy()
    obs_positions = np.where(mask)
    n_obs = len(obs_positions[0])

    if n_obs > 0:
        data_std = np.std(y_true[obs_positions])
        gaussian_noise = np.random.normal(0, data_std * noise_level * 0.7, size=n_obs)
        y_noisy[obs_positions] += gaussian_noise

        n_outliers = int(n_obs * noise_level * 0.3)
        if n_outliers > 0:
            outlier_indices = np.random.choice(n_obs, n_outliers, replace=False)
            data_mean = np.mean(y_true[obs_positions])

            for idx in outlier_indices:
                i, j = obs_positions[0][idx], obs_positions[1][idx]
                outlier_magnitude = np.random.uniform(4, 6) * data_std
                sign = 1 if np.random.rand() > 0.5 else -1
                y_noisy[i, j] = data_mean + sign * outlier_magnitude

    return y_noisy