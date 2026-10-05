import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import join
from os import environ
#from torchmetrics.functional import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda:0")
#torch.set_num_threads(2)

home=environ['HOME']
folder1='Alternative_trainings'
folder2='Datasets_20'


Condition_inputs=pd.read_csv(join(home,folder1, folder2,'T_Sampling_uniform_20.csv'))

name_11='State_space_cte_pressure_T' 
name_12='_phi_'
name_21='Reaction_rates_cte_pressure_T'
name_22='_phi_'

end='.csv'

phi_name=1.0

n_autoregression=1

epochs=15000

autosave=1000

def hydrogen_log_transform_sandiego_cantera(cantera_species):
    
    cantera_time=cantera_species.iloc[:,1]
    cantera_temperature=cantera_species.iloc[:,2]
    cantera_pressure=cantera_species.iloc[:,3]
    
    cantera_species_fractions=cantera_species.iloc[:,4:12]
    
    n_columns_mass_fraction=np.shape(cantera_species_fractions)[1]

    #cantera_species_fractions.apply(lambda x:(x**BCT-1)/BCT)
    #cantera_species_fractions=np.power(cantera_species_fractions,BCT)-1
    #cantera_species_fractions=cantera_species_fractions/BCT

    data_all=pd.concat([cantera_temperature, cantera_species_fractions], axis=1)
    
    data_all=np.log(data_all+1e-8)

    columns=data_all.columns.to_list()
    
    return data_all, n_columns_mass_fraction, columns

def z_score_data_transformation_time_shift(dataset, n_autoregression, columns):

    # Calculate Z-scores
    iterations = np.shape(dataset)[1]
    means = np.zeros(iterations)
    stds = np.zeros(iterations)

    for j in range(iterations):
        means[j] = np.mean(dataset.iloc[:, j])
        stds[j] = np.std(dataset.iloc[:, j])

        dataset.iloc[:, j] = (dataset.iloc[:, j] - means[j]) / stds[j]

    # If we have more than 1 autoregression, we need to shift the dataset
    if n_autoregression != 1:
        dataset_copy = dataset.copy()
        
        # Create the first shifted dataset with proper indexing
        core_data=dataset_copy.iloc[1:-n_autoregression, 1:].reset_index(drop=True, inplace=True)
        shifted_dfs = [core_data]

        for i in range(1, n_autoregression + 2):
            # Shift the dataset
            dataset_copy2 = dataset.copy()
            shifted_df = dataset_copy2.iloc[i:-n_autoregression+1, :]

            # Reset the index after shifting to keep data clean
            shifted_df.reset_index(drop=True, inplace=True)
            
            shifted_dfs.append(shifted_df)
        
        # Concatenate all the shifted data frames together
        result_df = pd.concat(shifted_dfs, axis=1)

        # Now your result_df should have all the time-shifted data in separate columns
        dataset_out = result_df
        dataset_out=dataset_out.iloc[:-n_autoregression,:]

    else:
        # For the case when autoregression is 1 (i.e., no shift)
        dataset_t0 = dataset.iloc[:-1, :]
        dataset_shift = dataset.iloc[1:, :]

        # Adding suffix to the shifted dataset columns for clarity
        dataset_shift = dataset_shift.add_suffix('_shift')

        # Reset index to ensure proper concatenation
        dataset_t0 = dataset_t0.reset_index(drop=True)
        dataset_shift = dataset_shift.reset_index(drop=True)

        # Concatenate the original data and the shifted data
        dataset_out = pd.concat([dataset_t0, dataset_shift], axis=1)

    return dataset_out, means, stds

iterations=np.shape(Condition_inputs)[0]
for i in range(iterations):
    cantera_species=pd.read_csv(join(home,folder1,folder2,name_11+str(round(Condition_inputs.loc[i,'T[K]'],3))+name_12+str(phi_name)+end))
    cantera_species=pd.DataFrame(cantera_species)
    
    dataset, n_columns_mass_fraction, columns=hydrogen_log_transform_sandiego_cantera(cantera_species)
    
    if i==0:
        n_samples=np.shape(dataset)[0]
        dataset_copy=dataset
    else:
        dataset_copy=pd.concat([dataset_copy,dataset],axis=0)

dataset_out, means, stds=z_score_data_transformation_time_shift(dataset_copy,n_autoregression, columns)

min_val=np.min(dataset_out)-0.1
max_val=np.max(dataset_out)+0.1 

seed=42
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_float32_matmul_precision('medium')

alpha=0.001

dropout_p=0 #dropout layer percentage

input_size=n_columns_mass_fraction+1 #the number of mass fractions plus one more column for the temperature

output_size=input_size #we want the same shape of the input vector 

l1=300
l2=l1
l3=l1


architecture_e=[l1,l2,l3]

class NN(nn.Module): #The part inside the parenthesis is a standard syntaxis function for defining the class object
    
    def __init__(self,input_size,output_size,dropout_p, architecture_e,min_val,max_val, alpha):
        super(NN,self).__init__()
        
        self.input_size=input_size
        self.output_size=output_size
        self.dropout_p=dropout_p
        self.architecture_e=architecture_e
        self.min_val=min_val
        self.max_val=max_val
        self.alpha=alpha
        
        encoder=[]
        
        if len(architecture_e)==1:
                layer=nn.Linear(input_size,architecture_e[0])
                nn.init.xavier_normal_(layer.weight)
                encoder.append(layer)
                encoder.append(nn.Sigmoid())
                #encoder.append(nn.LeakyReLU(negative_slope=alpha))
                #encoder.append(nn.Hardtanh(min_val=min_val, max_val=max_val))
                encoder.append(nn.Dropout(p=dropout_p))
                
                layer=nn.Linear(architecture_e[0],output_size)
                nn.init.xavier_normal_(layer.weight)
                encoder.append(layer)
                encoder.append(nn.Sigmoid())
                #encoder.append(nn.LeakyReLU(negative_slope=alpha))
                #encoder.append(nn.Hardtanh(min_val=min_val, max_val=max_val))
                encoder.append(nn.Dropout(p=dropout_p))
        else: 
            for i in range(len(architecture_e)):
                if i==0:
                    layer=nn.Linear(input_size,architecture_e[i])
                    nn.init.xavier_normal_(layer.weight)
                    encoder.append(layer)
                    encoder.append(nn.Sigmoid())
                    #encoder.append(nn.LeakyReLU(negative_slope=alpha))
                    #encoder.append(nn.Hardtanh(min_val=min_val, max_val=max_val))
                    encoder.append(nn.Dropout(p=dropout_p))
                else:
                    layer=nn.Linear(architecture_e[i-1],architecture_e[i])
                    nn.init.xavier_normal_(layer.weight)
                    encoder.append(layer)
                    encoder.append(nn.Sigmoid())
                    #encoder.append(nn.LeakyReLU(negative_slope=alpha))
                    #encoder.append(nn.Hardtanh(min_val=min_val, max_val=max_val))
                    encoder.append(nn.Dropout(p=dropout_p))
            
            encoder.append(nn.Linear(architecture_e[-1],output_size)) #reduced size output
        
        self.encoder=nn.Sequential(*encoder)
        
    def forward(self,x):
        #x = x.astype('float32')
        #x=torch.from_numpy(x)
        output=self.encoder(x)
        #l1_reg=torch.abs(self.weights).sum()

        return output #, l1_reg


model=NN(input_size,output_size,dropout_p, architecture_e,min_val,max_val,alpha)

optimizer=torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999))

def loss_MSE(y_pred, y_train):
        
    # MSE loss function
    
    mse = torch.mean(torch.square(y_train-y_pred))

    loss = mse 
    return loss 

import os
from pathlib import Path
from os.path import join
from os.path import isfile
from pathlib import Path

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

early_stopping = EarlyStopping(tolerance=5, min_delta=10)

n_repetitions=1
repetitions=range(n_repetitions)


output_start=1+n_columns_mass_fraction
output_end=output_start+n_columns_mass_fraction+1
    
Inputs = dataset_out.iloc[:, :1 + n_columns_mass_fraction].to_numpy()
Outputs = dataset_out.iloc[:, output_start:output_end].to_numpy()

monitor = 'loss'

os.mkdir(monitor + '_layers-' + str(len(architecture_e)))

from sklearn.model_selection import RepeatedKFold

for j in range(len(repetitions)):
    n_split = 5
    n_repeats = repetitions[j] + 1

    counting = 0

    directory_1 = monitor + '_layers-' + str(len(architecture_e))
    directory_2 = 'repeats_' + str(repetitions[j] + 1)
    address = join(directory_1, directory_2)
    os.mkdir(address)

    for train_index, test_index in RepeatedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=42).split(Inputs):
        x_train, x_test = Inputs[train_index], Inputs[test_index]
        y_train, y_test = Outputs[train_index], Outputs[test_index]

        Autoencoder = model
        Autoencoder = Autoencoder.to(device)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train = torch.from_numpy(x_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)

        final_losses = []
        reg_losses = []

        for k in range(epochs):
            k = k + 1

            y_pred = Autoencoder.forward(x_train)

            loss = loss_MSE(y_pred, torch.from_numpy(y_train).to(device))

            val_y_pred = Autoencoder.forward(x_test)

            val_loss = loss_MSE(val_y_pred, torch.from_numpy(y_test).to(device))

            final_losses.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            early_stopping(loss, val_loss)

            if early_stopping.early_stop:
                print("We are at epoch:", i)
                break

        name = join(address, f'k_model_' + str(counting) + '.pth')
        target = Path(address)
        torch.save(Autoencoder, name)

        name2 = join(address, f'k_model_loss' + str(counting) + '.csv')
        #name3 = join(address, f'k_model_reg_loss' + str(counting) + '.csv')

        final_losses = np.asarray(final_losses)
        #reg_losses = np.asarray(reg_losses)

        np.savetxt(name2, final_losses, delimiter=',')
        #np.savetxt(name3, reg_losses, delimiter=',')

        counting = counting + 1











