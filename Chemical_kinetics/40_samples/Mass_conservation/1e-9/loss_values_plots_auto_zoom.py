import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import join
from os import environ
from pathlib import Path
import re

# Automatically detect the available K_model_n.pth model (n = 0, ..., 4)
model_candidates = []
for model_path in Path(".").glob("K_model_*.pth"):
    match = re.fullmatch(r"K_model_([0-4])\.pth", model_path.name)
    if match:
        model_candidates.append((int(match.group(1)), model_path))

# Accept lowercase k as well, for compatibility with older runs.
if not model_candidates:
    for model_path in Path(".").glob("k_model_*.pth"):
        match = re.fullmatch(r"k_model_([0-4])\.pth", model_path.name)
        if match:
            model_candidates.append((int(match.group(1)), model_path))

model_candidates.sort(key=lambda item: item[0])

if not model_candidates:
    raise FileNotFoundError(
        "No model file found. Expected K_model_n.pth with n between 0 and 4."
    )

if len(model_candidates) > 1:
    raise RuntimeError(
        "More than one K_model_n.pth file was found: "
        + ", ".join(str(path) for _, path in model_candidates)
        + ". Keep only the model you want to evaluate in this folder."
    )

model_number, model_file = model_candidates[0]
print(f"Detected model: {model_file.name} (n = {model_number})")

# Publication-style formatting
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Automatic zoom settings
zoom_tail = 500          # use the last N epochs to determine the zoom
zoom_percentile = 99.0   # ignore rare spikes when setting the upper limit
zoom_padding = 0.10      # add 10% headroom above the selected range

def automatic_zoom_ylim(*arrays, tail=zoom_tail, percentile=zoom_percentile, padding=zoom_padding):
    """Return an automatic y-axis range for loss zoom plots.

    The range is estimated from the final `tail` samples of all supplied
    arrays. A percentile is used instead of the absolute maximum so that an
    isolated spike does not destroy the useful zoom level.
    """
    values = []
    for array in arrays:
        a = np.asarray(array, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size:
            values.append(a[-min(tail, a.size):])

    if not values:
        return 0.0, 1.0

    values = np.concatenate(values)
    upper = np.percentile(values, percentile)
    lower = np.min(values)

    # Losses should normally be non-negative; keep zero visible when suitable.
    if lower >= 0:
        lower = 0.0

    span = upper - lower
    if span <= 0:
        span = max(abs(upper), 1e-12)

    return lower, upper + padding * span

i = model_number

loss=np.genfromtxt('k_model_loss'+str(i)+'.csv',
                delimiter=",", dtype=float) 
reg_losses=np.genfromtxt('k_model_reg_loss'+str(i)+'.csv',
                delimiter=",", dtype=float)

losses=loss+reg_losses

array_lenght=len(loss)

average_start=array_lenght-500

print('mse loss average', np.mean(loss[average_start:]))
print('mass conservation avereage', np.mean(reg_losses[average_start:]))
print('total loss average', np.mean(losses[average_start:]))

loss_mean_values=np.transpose([np.mean(loss[average_start:]),np.mean(reg_losses[average_start:]),np.mean(losses[average_start:])])
print(np.shape(loss_mean_values))
mean_values=pd.DataFrame(loss_mean_values, index=['mse','reg','total'])
mean_values.to_csv('loss_mean_values.csv')

plt.figure(1,figsize=(13,9),dpi=400)
plt.plot(loss, label=r'$ \mathcal{L}_{mse} $')
plt.xlabel('Epochs',fontsize=30)
plt.ylabel('Loss', fontsize=30, rotation='horizontal', labelpad=35.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.savefig(join('loss_graph.png'),dpi=600,bbox_inches='tight')
plt.clf()

plt.figure(1,figsize=(13,9),dpi=400)
plt.plot(loss, label=r'$ \mathcal{L}_{mse} $')
plt.xlabel('Epochs',fontsize=30)
plt.ylabel('Loss', fontsize=30, rotation='horizontal', labelpad=35.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
ymin, ymax = automatic_zoom_ylim(loss)
plt.ylim(ymin, ymax)
plt.savefig(join('loss_graph_zoom.png'),dpi=600,bbox_inches='tight')
plt.clf()

plt.figure(2,figsize=(13,9),dpi=400)
plt.plot(reg_losses, label=r'$ \mathcal{L}_{mass} $', color= 'crimson')
plt.xlabel('Epochs',fontsize=30)
plt.ylabel('Loss', fontsize=30, rotation='horizontal', labelpad=35.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.savefig(join('loss_regularizer_graph.png'),dpi=600,bbox_inches='tight')
plt.clf()

plt.figure(2,figsize=(13,9),dpi=400)
plt.plot(reg_losses, label=r'$ \mathcal{L}_{mass} $', color='crimson')
plt.xlabel('Epochs',fontsize=30)
plt.ylabel('Loss', fontsize=30, rotation='horizontal', labelpad=35.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
ymin, ymax = automatic_zoom_ylim(reg_losses)
plt.ylim(ymin, ymax)
plt.savefig(join('loss_regularizer_graph_zoom.png'),dpi=600,bbox_inches='tight')
plt.clf()

plt.figure(3,figsize=(13,9),dpi=400)
plt.plot(loss, label=r'$ \mathcal{L}_{mse} $')
plt.plot(reg_losses, label=r'$ \mathcal{L}_{mass} $', color='crimson')
plt.plot(losses, label=r'$ \mathcal{L} $', color='k')
plt.xlabel('Epochs',fontsize=30)
plt.ylabel('Loss', fontsize=30, rotation='horizontal', labelpad=35.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.savefig(join('total_loss_graph.png'),dpi=600,bbox_inches='tight')
plt.clf()

plt.figure(3,figsize=(13,9),dpi=400)
plt.plot(loss, label=r'$ \mathcal{L}_{mse} $')
plt.plot(reg_losses, label=r'$ \mathcal{L}_{mass} $', color='crimson')
plt.plot(losses, label=r'$ \mathcal{L} $', color='k')
plt.xlabel('Epochs',fontsize=30)
plt.ylabel('Loss', fontsize=30, rotation='horizontal', labelpad=35.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
ymin, ymax = automatic_zoom_ylim(loss, reg_losses, losses)
plt.ylim(ymin, ymax)
plt.savefig(join('total_loss_graph_zoom.png'),dpi=600,bbox_inches='tight')
plt.clf()
