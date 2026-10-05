import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import join
from os import environ
import os
from pathlib import Path
from sklearn.model_selection import RepeatedKFold

# from torchmetrics.functional import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")
# torch.set_num_threads(2)

home = environ['HOME']
folder1 = 'Alternative_trainings'
folder2 = 'Datasets_30'

Condition_inputs = pd.read_csv(join(home, folder1, folder2, 'T_Sampling_uniform_30.csv'))

name_11 = 'State_space_cte_pressure_T'
name_12 = '_phi_'
name_21 = 'Reaction_rates_cte_pressure_T'
name_22 = '_phi_'

end = '.csv'

phi_name = 1.0

n_autoregression = 1

epochs = 15000

autosave = 1000

grad_eps = 1e-12
use_global_norm_ratio = False

def hydrogen_log_transform_sandiego_cantera(cantera_species):
    cantera_time = cantera_species.iloc[:, 1]
    cantera_temperature = cantera_species.iloc[:, 2]
    cantera_pressure = cantera_species.iloc[:, 3]

    cantera_species_fractions = cantera_species.iloc[:, 4:12]

    n_columns_mass_fraction = np.shape(cantera_species_fractions)[1]

    data_all = pd.concat([cantera_temperature, cantera_species_fractions], axis=1)
    data_all = np.log(data_all + 1e-8)

    columns = data_all.columns.to_list()

    return data_all, n_columns_mass_fraction, columns


def z_score_data_transformation_time_shift(dataset, n_autoregression, columns):
    iterations = np.shape(dataset)[1]
    means = np.zeros(iterations)
    stds = np.zeros(iterations)

    for j in range(iterations):
        means[j] = np.mean(dataset.iloc[:, j])
        stds[j] = np.std(dataset.iloc[:, j])
        dataset.iloc[:, j] = (dataset.iloc[:, j] - means[j]) / stds[j]

    if n_autoregression != 1:
        dataset_copy = dataset.copy()
        core_data = dataset_copy.iloc[1:-n_autoregression, 1:].reset_index(drop=True)
        shifted_dfs = [core_data]

        for i in range(1, n_autoregression + 2):
            dataset_copy2 = dataset.copy()
            shifted_df = dataset_copy2.iloc[i:-n_autoregression + 1, :]
            shifted_df = shifted_df.reset_index(drop=True)
            shifted_dfs.append(shifted_df)

        result_df = pd.concat(shifted_dfs, axis=1)
        dataset_out = result_df
        dataset_out = dataset_out.iloc[:-n_autoregression, :]
    else:
        dataset_t0 = dataset.iloc[:-1, :]
        dataset_shift = dataset.iloc[1:, :]
        dataset_shift = dataset_shift.add_suffix('_shift')
        dataset_t0 = dataset_t0.reset_index(drop=True)
        dataset_shift = dataset_shift.reset_index(drop=True)
        dataset_out = pd.concat([dataset_t0, dataset_shift], axis=1)

    return dataset_out, means, stds


iterations = np.shape(Condition_inputs)[0]
for i in range(iterations):
    cantera_species = pd.read_csv(
        join(home, folder1, folder2, name_11 + str(round(Condition_inputs.loc[i, 'T[K]'], 3)) + name_12 + str(phi_name) + end)
    )
    cantera_species = pd.DataFrame(cantera_species)

    dataset, n_columns_mass_fraction, columns = hydrogen_log_transform_sandiego_cantera(cantera_species)

    if i == 0:
        n_samples = np.shape(dataset)[0]
        dataset_copy = dataset
    else:
        dataset_copy = pd.concat([dataset_copy, dataset], axis=0)

dataset_out, means, stds = z_score_data_transformation_time_shift(dataset_copy, n_autoregression, columns)

min_val = np.min(dataset_out) - 0.1
max_val = np.max(dataset_out) + 0.1

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_float32_matmul_precision('medium')

alpha = 0.001

dropout_p = 0

input_size = n_columns_mass_fraction + 1
output_size = input_size

l1 = 300
l2 = l1
l3 = l1

architecture_e = [l1, l2, l3]

means_tensor = torch.tensor(means, dtype=torch.float32, device=device)
stds_tensor = torch.tensor(stds, dtype=torch.float32, device=device)


class NN(nn.Module):
    def __init__(self, input_size, output_size, dropout_p, architecture_e, min_val, max_val):
        super(NN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.architecture_e = architecture_e
        self.min_val = min_val
        self.max_val = max_val

        encoder = []

        if len(architecture_e) == 1:
            layer = nn.Linear(input_size, architecture_e[0])
            nn.init.xavier_normal_(layer.weight)
            encoder.append(layer)
            encoder.append(nn.Sigmoid())
            encoder.append(nn.Dropout(p=dropout_p))

            layer = nn.Linear(architecture_e[0], output_size)
            nn.init.xavier_normal_(layer.weight)
            encoder.append(layer)
            encoder.append(nn.Sigmoid())
            encoder.append(nn.Dropout(p=dropout_p))
        else:
            for i in range(len(architecture_e)):
                if i == 0:
                    layer = nn.Linear(input_size, architecture_e[i])
                else:
                    layer = nn.Linear(architecture_e[i - 1], architecture_e[i])
                nn.init.xavier_normal_(layer.weight)
                encoder.append(layer)
                encoder.append(nn.Sigmoid())
                encoder.append(nn.Dropout(p=dropout_p))

            encoder.append(nn.Linear(architecture_e[-1], output_size))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        output = self.encoder(x)
        return output


model = NN(input_size, output_size, dropout_p, architecture_e, min_val, max_val)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def loss_MSE(y_pred, y_train, means_t, stds_t): #reg_weight=1e-9):
    mse = torch.mean(torch.square(y_train - y_pred))

    y_pred_rescaled = y_pred[:, 1:] * stds_t[1:] + means_t[1:]
    y_true_rescaled = y_train[:, 1:] * stds_t[1:] + means_t[1:]

    y_pred_rescaled = torch.exp(y_pred_rescaled)
    y_true_rescaled = torch.exp(y_true_rescaled)

    reg = torch.square(torch.sum(y_true_rescaled, dim=1) - torch.sum(y_pred_rescaled, dim=1)).mean()
    #reg = reg_weight * reg

    return mse.float(), reg.float()


def grad_norm(grad, norm_type=2):
    return torch.norm(grad, p=norm_type)


def global_grad_norm(grads, norm_type=2, device_override=None):
    valid = [g.reshape(-1) for g in grads if g is not None]
    if not valid:
        dev = device_override if device_override is not None else device
        return torch.tensor(0.0, device=dev)
    flat = torch.cat(valid)
    return torch.norm(flat, p=norm_type)


def compute_separate_grads(loss_mse, loss_reg, model):
    params = [p for p in model.parameters() if p.requires_grad]

    mse_grads = torch.autograd.grad(
        loss_mse,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    reg_grads = torch.autograd.grad(
        loss_reg,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    return params, mse_grads, reg_grads


def assign_combined_grads_with_norm_ratio(params, mse_grads, reg_grads, norm_type=2, eps=1e-12, use_global=False):
    grad_stats = {}

    global_scale = None
    global_mse_norm = None
    global_reg_norm = None

    if use_global:
        ref_device = None
        for p in params:
            ref_device = p.device
            break
        global_mse_norm = global_grad_norm(mse_grads, norm_type=norm_type, device_override=ref_device)
        global_reg_norm = global_grad_norm(reg_grads, norm_type=norm_type, device_override=ref_device)
        if global_reg_norm > global_mse_norm:
            global_scale = global_mse_norm / (global_reg_norm + eps)
        else:
            global_scale = torch.tensor(1.0, device=global_mse_norm.device, dtype=global_mse_norm.dtype)

    for idx, (p, g_mse, g_reg) in enumerate(zip(params, mse_grads, reg_grads)):
        if g_mse is None and g_reg is None:
            p.grad = None
            continue

        if g_mse is None:
            g_mse = torch.zeros_like(p)
        if g_reg is None:
            g_reg = torch.zeros_like(p)

        mse_norm = grad_norm(g_mse, norm_type=norm_type)
        reg_norm = grad_norm(g_reg, norm_type=norm_type)

        if use_global:
            scale = global_scale
            scaled = bool(global_reg_norm > global_mse_norm)
        else:
            if reg_norm > mse_norm:
                scale = mse_norm / (reg_norm + eps)
                scaled = True
            else:
                scale = torch.tensor(1.0, device=g_reg.device, dtype=g_reg.dtype)
                scaled = False

        g_reg_scaled = g_reg * scale
        p.grad = g_mse + g_reg_scaled

        grad_stats[f'param_{idx}'] = {
            'mse_norm': float(mse_norm.detach().cpu().item()),
            'reg_norm': float(reg_norm.detach().cpu().item()),
            'scale': float(scale.detach().cpu().item()),
            'scaled': scaled,
        }

    return grad_stats


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        train_loss = float(train_loss.detach().cpu().item()) if torch.is_tensor(train_loss) else float(train_loss)
        validation_loss = float(validation_loss.detach().cpu().item()) if torch.is_tensor(validation_loss) else float(validation_loss)
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


n_repetitions = 1
repetitions = range(n_repetitions)

if n_autoregression == 1:
    output_start = 1 + n_columns_mass_fraction
    output_end = output_start + n_columns_mass_fraction + 1

    Inputs = dataset_out.iloc[:, 0:1 + n_columns_mass_fraction].to_numpy()
    Outputs = dataset_out.iloc[:, output_start:output_end].to_numpy()

    monitor = 'loss'

    os.mkdir(monitor + '_layers-' + str(len(architecture_e)))

    for j in range(len(repetitions)):
        n_split = 5
        n_repeats = repetitions[j] + 1

        counting = 0

        directory_1 = monitor + '_layers-' + str(len(architecture_e))
        directory_2 = 'repeats_' + str(repetitions[j] + 1)
        address = join(directory_1, directory_2)
        os.mkdir(address)

        for train_index, test_index in RepeatedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=42).split(Inputs):
            early_stopping = EarlyStopping(tolerance=5, min_delta=10)

            x_train, x_test = Inputs[train_index], Inputs[test_index]
            y_train, y_test = Outputs[train_index], Outputs[test_index]

            Autoencoder = model
            Autoencoder = Autoencoder.to(device)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')

            x_train = torch.from_numpy(x_train).to(device)
            x_test = torch.from_numpy(x_test).to(device)
            y_train_t = torch.from_numpy(y_train).to(device)
            y_test_t = torch.from_numpy(y_test).to(device)

            final_losses = []
            reg_losses = []
            grad_mse_norms = []
            grad_reg_norms = []
            grad_scales = []
            scaled_flags = []

            for k in range(epochs):
                k = k + 1

                y_pred = Autoencoder.forward(x_train)
                mse, reg = loss_MSE(y_pred, y_train_t, means_tensor, stds_tensor)

                val_y_pred = Autoencoder.forward(x_test)
                mse_val_loss, val_reg_loss = loss_MSE(val_y_pred, y_test_t, means_tensor, stds_tensor)

                final_losses.append(float(mse.detach().cpu().item()))
                reg_losses.append(float(reg.detach().cpu().item()))

                val_loss = mse_val_loss + val_reg_loss

                optimizer.zero_grad()
                params, mse_grads, reg_grads = compute_separate_grads(mse, reg, Autoencoder)
                grad_stats = assign_combined_grads_with_norm_ratio(
                    params,
                    mse_grads,
                    reg_grads,
                    norm_type=2,
                    eps=grad_eps,
                    use_global=use_global_norm_ratio,
                )
                optimizer.step()

                fold_mse_norm = float(np.mean([v['mse_norm'] for v in grad_stats.values()])) if grad_stats else 0.0
                fold_reg_norm = float(np.mean([v['reg_norm'] for v in grad_stats.values()])) if grad_stats else 0.0
                fold_scale = float(np.mean([v['scale'] for v in grad_stats.values()])) if grad_stats else 1.0
                fold_scaled = int(any(v['scaled'] for v in grad_stats.values())) if grad_stats else 0

                grad_mse_norms.append(fold_mse_norm)
                grad_reg_norms.append(fold_reg_norm)
                grad_scales.append(fold_scale)
                scaled_flags.append(fold_scaled)

                early_stopping(mse + reg, val_loss)

                if k % autosave == 0:
                    name = join(address, f'k_model_epochs_{k}_{counting}.pth')
                    torch.save(Autoencoder, name)

                    np.savetxt(join(address, f'k_model_epochs_{k}_loss{counting}.csv'), np.asarray(final_losses), delimiter=',')
                    np.savetxt(join(address, f'k_model__epochs_{k}_reg_loss{counting}.csv'), np.asarray(reg_losses), delimiter=',')
                    grad_df = pd.DataFrame({
                        'epoch': np.arange(1, len(grad_mse_norms) + 1),
                        'mse_grad_norm': grad_mse_norms,
                        'reg_grad_norm': grad_reg_norms,
                        'reg_scale': grad_scales,
                        'reg_scaled': scaled_flags,
                    })
                    grad_df.to_csv(join(address, f'k_model_epochs_{k}_grad_stats{counting}.csv'), index=False)

                if early_stopping.early_stop:
                    print("We are at epoch:", k)
                    break

            torch.save(Autoencoder, join(address, f'k_model_{counting}.pth'))
            np.savetxt(join(address, f'k_model_loss{counting}.csv'), np.asarray(final_losses), delimiter=',')
            np.savetxt(join(address, f'k_model_reg_loss{counting}.csv'), np.asarray(reg_losses), delimiter=',')
            pd.DataFrame({
                'epoch': np.arange(1, len(grad_mse_norms) + 1),
                'mse_grad_norm': grad_mse_norms,
                'reg_grad_norm': grad_reg_norms,
                'reg_scale': grad_scales,
                'reg_scaled': scaled_flags,
            }).to_csv(join(address, f'k_model_grad_stats{counting}.csv'), index=False)

            counting = counting + 1
else:
    output_start = 1 + n_columns_mass_fraction
    output_end = output_start + n_columns_mass_fraction + 1

    Inputs = dataset_copy.iloc[:, :1 + n_columns_mass_fraction].to_numpy()
    Outputs = dataset_copy.iloc[:, output_start:output_end].to_numpy()

    monitor = 'loss'

    os.mkdir(monitor + '_layers-' + str(len(architecture_e)))

    for j in range(len(repetitions)):
        n_split = 5
        n_repeats = repetitions[j] + 1

        counting = 0

        directory_1 = monitor + '_layers-' + str(len(architecture_e))
        directory_2 = 'repeats_' + str(repetitions[j] + 1)
        address = join(directory_1, directory_2)
        os.mkdir(address)

        for train_index, test_index in RepeatedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=42).split(Inputs):
            early_stopping = EarlyStopping(tolerance=5, min_delta=10)

            x_train, x_test = Inputs[train_index], Inputs[test_index]
            y_train, y_test = Outputs[train_index], Outputs[test_index]

            Autoencoder = model
            Autoencoder = Autoencoder.to(device)

            x_train = torch.from_numpy(x_train.astype('float32')).to(device)
            x_test = torch.from_numpy(x_test.astype('float32')).to(device)
            y_train_t = torch.from_numpy(y_train.astype('float32')).to(device)
            y_test_t = torch.from_numpy(y_test.astype('float32')).to(device)

            final_losses = []
            reg_losses = []
            grad_mse_norms = []
            grad_reg_norms = []
            grad_scales = []
            scaled_flags = []

            for k in range(epochs):
                k = k + 1

                y_pred = Autoencoder.forward(x_train)

                output_start = (n_columns_mass_fraction + 1)
                output_end = output_start + n_columns_mass_fraction + 1

                mse, reg = loss_MSE(y_pred, y_train_t[:, output_start:output_end], means_tensor, stds_tensor)
                mse_total = mse
                reg_total = reg

                val_y_pred = Autoencoder.forward(x_test)
                mse_val_loss, val_reg_loss = loss_MSE(val_y_pred, y_test_t[:, output_start:output_end], means_tensor, stds_tensor)
                val_mse_total = mse_val_loss
                val_reg_total = val_reg_loss

                for l in range(n_autoregression - 1):
                    output_start = l * (n_columns_mass_fraction + 1)
                    output_end = output_start + n_columns_mass_fraction + 1

                    y_pred = Autoencoder.forward(y_pred)
                    mse_step, reg_step = loss_MSE(y_pred, y_train_t[:, output_start:output_end], means_tensor, stds_tensor)
                    mse_total = mse_total + mse_step
                    reg_total = reg_total + reg_step

                    val_y_pred = Autoencoder.forward(val_y_pred)
                    mse_val_step, reg_val_step = loss_MSE(val_y_pred, y_test_t[:, output_start:output_end], means_tensor, stds_tensor)
                    val_mse_total = val_mse_total + mse_val_step
                    val_reg_total = val_reg_total + reg_val_step

                final_losses.append(float(mse_total.detach().cpu().item()))
                reg_losses.append(float(reg_total.detach().cpu().item()))

                optimizer.zero_grad()
                params, mse_grads, reg_grads = compute_separate_grads(mse_total, reg_total, Autoencoder)
                grad_stats = assign_combined_grads_with_norm_ratio(
                    params,
                    mse_grads,
                    reg_grads,
                    norm_type=2,
                    eps=grad_eps,
                    use_global=use_global_norm_ratio,
                )
                optimizer.step()

                fold_mse_norm = float(np.mean([v['mse_norm'] for v in grad_stats.values()])) if grad_stats else 0.0
                fold_reg_norm = float(np.mean([v['reg_norm'] for v in grad_stats.values()])) if grad_stats else 0.0
                fold_scale = float(np.mean([v['scale'] for v in grad_stats.values()])) if grad_stats else 1.0
                fold_scaled = int(any(v['scaled'] for v in grad_stats.values())) if grad_stats else 0

                grad_mse_norms.append(fold_mse_norm)
                grad_reg_norms.append(fold_reg_norm)
                grad_scales.append(fold_scale)
                scaled_flags.append(fold_scaled)

                early_stopping(mse_total + reg_total, val_mse_total + val_reg_total)

                if k % autosave == 0:
                    torch.save(Autoencoder, join(address, f'k_model_epochs_{k}_{counting}.pth'))
                    np.savetxt(join(address, f'k_model_epochs_{k}_loss{counting}.csv'), np.asarray(final_losses), delimiter=',')
                    np.savetxt(join(address, f'k_model__epochs_{k}_reg_loss{counting}.csv'), np.asarray(reg_losses), delimiter=',')
                    pd.DataFrame({
                        'epoch': np.arange(1, len(grad_mse_norms) + 1),
                        'mse_grad_norm': grad_mse_norms,
                        'reg_grad_norm': grad_reg_norms,
                        'reg_scale': grad_scales,
                        'reg_scaled': scaled_flags,
                    }).to_csv(join(address, f'k_model_epochs_{k}_grad_stats{counting}.csv'), index=False)

                if early_stopping.early_stop:
                    print("We are at epoch:", k)
                    break

            torch.save(Autoencoder, join(address, f'k_model_{counting}.pth'))
            np.savetxt(join(address, f'k_model_loss{counting}.csv'), np.asarray(final_losses), delimiter=',')
            np.savetxt(join(address, f'k_model_reg_loss{counting}.csv'), np.asarray(reg_losses), delimiter=',')
            pd.DataFrame({
                'epoch': np.arange(1, len(grad_mse_norms) + 1),
                'mse_grad_norm': grad_mse_norms,
                'reg_grad_norm': grad_reg_norms,
                'reg_scale': grad_scales,
                'reg_scaled': scaled_flags,
            }).to_csv(join(address, f'k_model_grad_stats{counting}.csv'), index=False)

            counting = counting + 1
