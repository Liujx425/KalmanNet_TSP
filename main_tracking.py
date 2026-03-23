"""
Main entry point for LGA-KalmanNet Single Target Tracking.

Compares:
1. Observation Noise Floor
2. Standard Kalman Filter
3. Vanilla KalmanNet
4. LGA-KalmanNet (robust)

Supports testing under various outlier contamination levels.
"""

import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen
import Simulations.config as config
from Simulations.Target_Tracking.parameters import (
    F, H, Q_structure, R_structure, m, n, m1_0,
    add_outliers_to_observations
)

from Filters.KalmanFilter_test import KFTest
from KNet.KalmanNet_nn import KalmanNetNN
from KNet.LGA_KalmanNet_nn import LGA_KalmanNetNN
from Pipelines.Pipeline_EKF import Pipeline_EKF
from Pipelines.Pipeline_LGA import Pipeline_LGA

print("=" * 60)
print("LGA-KalmanNet Single Target Tracking")
print("=" * 60)

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet/'

####################
### Design Model ###
####################
args = config.general_settings()

### Dataset parameters ###
args.N_E = 1000   # training sequences
args.N_CV = 100    # validation sequences
args.N_T = 200     # test sequences

# Initial conditions
args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True
args.variance = 10  # initial state variance
args.distribution = 'normal'
m2_0 = args.variance * torch.eye(m)

# Sequence length
args.T = 100
args.T_test = 100
args.randomLength = False
train_lengthMask = None
cv_lengthMask = None
test_lengthMask = None

# Noise parameters
q2 = torch.tensor([0.1])   # process noise variance
r2 = torch.tensor([1.0])   # observation noise variance
print("Process noise q2:", q2.item())
print("Observation noise r2:", r2.item())
print("1/r2 [dB]:", (10 * torch.log10(1 / r2[0])).item())
print("1/q2 [dB]:", (10 * torch.log10(1 / q2[0])).item())

### Training parameters ###
args.use_cuda = True
args.n_steps = 2000
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-4
args.CompositionLoss = False
args.alpha = 0.3

### KalmanNet settings ###
args.in_mult_KNet = 5
args.out_mult_KNet = 40

### LGA-KalmanNet settings ###
args.lga_window = 5       # observation buffer window size
args.lga_d_model = 16     # LGA embedding dimension
args.lga_n_heads = 2      # number of LGA attention heads
args.lga_d_G = 32         # LGA metric embedding dimension
args.lga_eps = 1e-2       # LGA numerical stability
args.lga_weight = 0.1     # weight for LGA metric learning loss

### Outlier testing parameters ###
outlier_ratios = [0.0, 0.05, 0.1, 0.2, 0.3]
outlier_amplitude = 15.0   # amplitude of outlier spikes

if args.use_cuda:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        print("No GPU found, falling back to CPU")
        args.use_cuda = False
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
    print("Using CPU")

##########################
### Build System Model ###
##########################
Q = q2 * Q_structure
R = r2 * R_structure
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

print("\nSystem Model:")
print(f"  State dim: {m} (px, py, vx, vy)")
print(f"  Obs dim: {n} (px, py)")
print("  State Evolution Matrix F:")
print(F)
print("  Observation Matrix H:")
print(H)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Target_Tracking/data/'
dataFileName = 'tracking_4x2_T100.pt'
import os
os.makedirs(dataFolderName, exist_ok=True)
os.makedirs(path_results, exist_ok=True)
print("\nGenerating data...")
DataGen(args, sys_model, dataFolderName + dataFileName)
print("Loading data...")
[train_input, train_target, cv_input, cv_target, test_input, test_target,
 train_init, cv_init, test_init] = torch.load(
    dataFolderName + dataFileName, map_location=device)

print(f"  Train set: {train_target.size()}")
print(f"  CV set: {cv_target.size()}")
print(f"  Test set: {test_target.size()}")

########################################
### Evaluate Observation Noise Floor ###
########################################
print("\n" + "=" * 50)
print("Observation Noise Floor")
print("=" * 50)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)
for i in range(args.N_T):
    # Compare H^+ @ y with x (pseudo-inverse reconstruction)
    MSE_obs_linear_arr[i] = loss_obs(test_input[i], H.to(device) @ test_target[i]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)
print(f"Observation Noise Floor: {MSE_obs_dB_avg.item():.2f} [dB]")

##############################
### Evaluate Kalman Filter ###
##############################
print("\n" + "=" * 50)
print("Standard Kalman Filter (clean data)")
print("=" * 50)
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(
    args, sys_model, test_input, test_target,
    randomInit=True, test_init=test_init)

###################################
### Vanilla KalmanNet Pipeline ###
###################################
print("\n" + "=" * 50)
print("Vanilla KalmanNet (baseline)")
print("=" * 50)
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)
print(f"Trainable parameters: {sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad)}")

KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet_tracking")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)

[MSE_cv_linear_epoch, MSE_cv_dB_epoch,
 MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(
    sys_model, cv_input, cv_target, train_input, train_target, path_results,
    randomInit=True, cv_init=cv_init, train_init=train_init)

print("\n--- KalmanNet Test (clean data) ---")
[MSE_test_linear_arr_knet, MSE_test_linear_avg_knet, MSE_test_dB_avg_knet,
 knet_out, RunTime_knet] = KalmanNet_Pipeline.NNTest(
    sys_model, test_input, test_target, path_results,
    randomInit=True, test_init=test_init)

# Save KalmanNet best model for outlier testing
torch.save(KalmanNet_Pipeline.model, path_results + 'best-model-knet-tracking.pt')
KalmanNet_Pipeline.save()

####################################
### LGA-KalmanNet Pipeline ###
####################################
print("\n" + "=" * 50)
print("LGA-KalmanNet (robust)")
print("=" * 50)
LGA_KNet_model = LGA_KalmanNetNN()
LGA_KNet_model.NNBuild(sys_model, args)
print(f"Trainable parameters: {sum(p.numel() for p in LGA_KNet_model.parameters() if p.requires_grad)}")

LGA_Pipeline = Pipeline_LGA(strTime, "KNet", "LGA_KalmanNet_tracking")
LGA_Pipeline.setssModel(sys_model)
LGA_Pipeline.setModel(LGA_KNet_model)
LGA_Pipeline.setTrainingParams(args)

[MSE_cv_linear_epoch_lga, MSE_cv_dB_epoch_lga,
 MSE_train_linear_epoch_lga, MSE_train_dB_epoch_lga,
 G_loss_epoch] = LGA_Pipeline.NNTrain(
    sys_model, cv_input, cv_target, train_input, train_target, path_results,
    randomInit=True, cv_init=cv_init, train_init=train_init)

print("\n--- LGA-KalmanNet Test (clean data) ---")
[MSE_test_linear_arr_lga, MSE_test_linear_avg_lga, MSE_test_dB_avg_lga,
 lga_knet_out, RunTime_lga] = LGA_Pipeline.NNTest(
    sys_model, test_input, test_target, path_results,
    randomInit=True, test_init=test_init)

torch.save(LGA_Pipeline.model, path_results + 'best-model-lga-tracking.pt')
LGA_Pipeline.save()

###########################################
### Robustness Evaluation with Outliers ###
###########################################
print("\n" + "=" * 60)
print("Robustness Evaluation: Outlier Contamination")
print("=" * 60)

results_kf = {}
results_knet = {}
results_lga = {}

for ratio in outlier_ratios:
    print(f"\n--- Outlier ratio: {ratio:.0%}, amplitude: {outlier_amplitude} ---")

    if ratio == 0.0:
        test_input_corrupted = test_input
    else:
        test_input_corrupted, outlier_masks = add_outliers_to_observations(
            test_input, outlier_ratio=ratio, outlier_amplitude=outlier_amplitude)

    # KF on corrupted data
    print("  Kalman Filter:")
    [MSE_arr, MSE_avg, MSE_dB, _] = KFTest(
        args, sys_model, test_input_corrupted, test_target,
        randomInit=True, test_init=test_init)
    results_kf[ratio] = MSE_dB.item()

    # KalmanNet on corrupted data
    print("  KalmanNet:")
    knet_res = KalmanNet_Pipeline.NNTest(
        sys_model, test_input_corrupted, test_target, path_results,
        randomInit=True, test_init=test_init,
        load_model=True, load_model_path=path_results + 'best-model-knet-tracking.pt')
    results_knet[ratio] = knet_res[2].item()

    # LGA-KalmanNet on corrupted data
    print("  LGA-KalmanNet:")
    lga_res = LGA_Pipeline.NNTest(
        sys_model, test_input_corrupted, test_target, path_results,
        randomInit=True, test_init=test_init,
        load_model=True, load_model_path=path_results + 'best-model-lga-tracking.pt')
    results_lga[ratio] = lga_res[2].item()

###########################
### Print Summary Table ###
###########################
print("\n" + "=" * 60)
print("SUMMARY: MSE [dB] at Different Outlier Ratios")
print("=" * 60)
print(f"{'Outlier %':>12} {'KF':>10} {'KalmanNet':>12} {'LGA-KNet':>12}")
print("-" * 48)
for ratio in outlier_ratios:
    print(f"{ratio:>11.0%} {results_kf[ratio]:>10.2f} {results_knet[ratio]:>12.2f} {results_lga[ratio]:>12.2f}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
