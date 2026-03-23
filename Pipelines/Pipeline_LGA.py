"""
Training and testing pipeline for LGA-KalmanNet.

Extends Pipeline_EKF with:
- LGA metric learning loss (G_loss)
- Combined loss: MSE_state + lga_weight * G_loss
- Outlier robustness evaluation
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_extended


class Pipeline_LGA:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps
        self.N_B = args.n_batch
        self.learningRate = args.lr
        self.weightDecay = args.wd
        self.alpha = args.alpha
        self.lga_weight = getattr(args, 'lga_weight', 0.1)

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weightDecay
        )

    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target,
                path_results, MaskOnState=False, randomInit=False,
                cv_init=None, train_init=None,
                train_lengthMask=None, cv_lengthMask=None):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        self.G_loss_epoch = torch.zeros([self.N_steps])

        if MaskOnState:
            mask = torch.tensor([True, False, False])
            if SysModel.m == 2:
                mask = torch.tensor([True, False])
            elif SysModel.m == 4:
                mask = torch.tensor([True, True, False, False])

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            self.model.train()
            self.model.batch_size = self.N_B
            self.model.init_hidden_KNet()

            # Init batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                y_training_batch[ii, :, :] = train_input[index]
                train_target_batch[ii, :, :] = train_target[index]
                ii += 1

            # Init Sequence
            if randomInit:
                train_init_batch = torch.empty([self.N_B, SysModel.m, 1]).to(self.device)
                ii = 0
                for index in n_e:
                    train_init_batch[ii, :, 0] = torch.squeeze(train_init[index])
                    ii += 1
                self.model.InitSequence(train_init_batch, SysModel.T)
            else:
                self.model.InitSequence(
                    SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_B, 1, 1),
                    SysModel.T)

            # Forward Computation
            G_loss_total = 0.0
            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = torch.squeeze(
                    self.model(torch.unsqueeze(y_training_batch[:, :, t], 2)))
                G_loss_total += self.model.get_G_loss()

            G_loss_avg = G_loss_total / SysModel.T

            # Compute Training Loss
            if self.args.CompositionLoss:
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
                for t in range(SysModel.T):
                    y_hat[:, :, t] = torch.squeeze(
                        SysModel.h(torch.unsqueeze(x_out_training_batch[:, :, t], 2)))

                if MaskOnState:
                    MSE_trainbatch_linear_LOSS = (
                        self.alpha * self.loss_fn(x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
                        + (1 - self.alpha) * self.loss_fn(y_hat[:, mask, :], y_training_batch[:, mask, :])
                    )
                else:
                    MSE_trainbatch_linear_LOSS = (
                        self.alpha * self.loss_fn(x_out_training_batch, train_target_batch)
                        + (1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
                    )
            else:
                if MaskOnState:
                    MSE_trainbatch_linear_LOSS = self.loss_fn(
                        x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
                else:
                    MSE_trainbatch_linear_LOSS = self.loss_fn(
                        x_out_training_batch, train_target_batch)

            # Combined loss: state MSE + LGA metric learning loss
            total_loss = MSE_trainbatch_linear_LOSS + self.lga_weight * G_loss_avg

            # Record losses
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            if isinstance(G_loss_avg, (int, float)):
                self.G_loss_epoch[ti] = G_loss_avg
            else:
                self.G_loss_epoch[ti] = G_loss_avg.item()

            # Backward + Optimize
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            #################################
            ### Validation Sequence Batch ###
            #################################
            self.model.eval()
            self.model.batch_size = self.N_CV
            self.model.init_hidden_KNet()

            with torch.no_grad():
                SysModel.T_test = cv_input.size()[-1]
                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)

                if randomInit:
                    if cv_init is None:
                        self.model.InitSequence(
                            SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_CV, 1, 1),
                            SysModel.T_test)
                    else:
                        self.model.InitSequence(cv_init, SysModel.T_test)
                else:
                    self.model.InitSequence(
                        SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_CV, 1, 1),
                        SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(
                        self.model(torch.unsqueeze(cv_input[:, :, t], 2)))

                if MaskOnState:
                    MSE_cvbatch_linear_LOSS = self.loss_fn(
                        x_out_cv_batch[:, mask, :], cv_target[:, mask, :])
                else:
                    MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results + 'best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti,
                  "MSE Training:", self.MSE_train_dB_epoch[ti].item(), "[dB]",
                  "MSE Validation:", self.MSE_cv_dB_epoch[ti].item(), "[dB]",
                  "G_loss:", self.G_loss_epoch[ti].item())

            if ti > 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training:", d_train.item(), "[dB]",
                      "diff MSE Validation:", d_cv.item(), "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt,
                  "Optimal:", self.MSE_cv_dB_opt.item(), "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch,
                self.MSE_train_linear_epoch, self.MSE_train_dB_epoch,
                self.G_loss_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results,
               MaskOnState=False, randomInit=False, test_init=None,
               load_model=False, load_model_path=None, test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device)
        else:
            self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

        if MaskOnState:
            mask = torch.tensor([True, False, False])
            if SysModel.m == 2:
                mask = torch.tensor([True, False])
            elif SysModel.m == 4:
                mask = torch.tensor([True, True, False, False])

        loss_fn = nn.MSELoss(reduction='mean')

        self.model.eval()
        self.model.batch_size = self.N_T
        self.model.init_hidden_KNet()
        torch.no_grad()

        start = time.time()

        if randomInit:
            self.model.InitSequence(test_init, SysModel.T_test)
        else:
            self.model.InitSequence(
                SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_T, 1, 1),
                SysModel.T_test)

        for t in range(0, SysModel.T_test):
            x_out_test[:, :, t] = torch.squeeze(
                self.model(torch.unsqueeze(test_input[:, :, t], 2)))

        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):
            if MaskOnState:
                self.MSE_test_linear_arr[j] = loss_fn(
                    x_out_test[j, mask, :], test_target[j, mask, :]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(
                    x_out_test[j, :, :], test_target[j, :, :]).item()

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = (
            10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg)
            - self.MSE_test_dB_avg
        )

        print(self.modelName + " - MSE Test:", self.MSE_test_dB_avg.item(), "[dB]")
        print(self.modelName + " - STD Test:", self.test_std_dB.item(), "[dB]")
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg,
                self.MSE_test_dB_avg, x_out_test, t]

    def NNTest_with_outliers(self, SysModel, test_input_clean, test_target,
                             path_results, outlier_ratios=[0.0, 0.05, 0.1, 0.2],
                             outlier_amplitude=10.0, randomInit=False,
                             test_init=None):
        """
        Test robustness across different outlier levels.
        Returns results dict: {outlier_ratio: (MSE_dB, STD_dB)}
        """
        from Simulations.Target_Tracking.parameters import add_outliers_to_observations

        results = {}
        for ratio in outlier_ratios:
            if ratio == 0.0:
                test_input = test_input_clean
            else:
                test_input, _ = add_outliers_to_observations(
                    test_input_clean, outlier_ratio=ratio,
                    outlier_amplitude=outlier_amplitude)

            res = self.NNTest(SysModel, test_input, test_target, path_results,
                              randomInit=randomInit, test_init=test_init)
            results[ratio] = (res[2].item(), self.test_std_dB.item())
            print(f"Outlier ratio {ratio:.0%}: MSE={res[2].item():.2f} dB, "
                  f"STD={self.test_std_dB.item():.2f} dB")

        return results

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):
        self.Plot = Plot_extended(self.folderName, self.modelName)
        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg,
                                self.MSE_cv_dB_epoch,
                                self.MSE_train_dB_epoch)
        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)
