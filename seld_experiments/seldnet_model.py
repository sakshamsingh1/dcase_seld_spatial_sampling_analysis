#
# The SELDnet architecture
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import math

class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss

# new TAEDPIT loss
class MSELoss_ADPIT_comb(object):
    def __init__(self,threshold=0.5,thr_loss_scale=1e-3):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')
        self.threshold = threshold
        self.epsilon = 1e-6
        self.thr_loss_scale = thr_loss_scale

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def _each_calc_angular(self, output, target):
        Oaz0, Taz0 = output[:, :, [0], :], target[:, :, [0], :]
        Oel0, Tel0 = output[:, :, [1], :], target[:, :, [1], :]
        Oaz1, Taz1 = output[:, :, [2], :], target[:, :, [2], :]
        Oel1, Tel1 = output[:, :, [3], :], target[:, :, [3], :]
        Oaz2, Taz2 = output[:, :, [4], :], target[:, :, [4], :]
        Oel2, Tel2 = output[:, :, [5], :], target[:, :, [5], :]

        #convert degree to radians
        PI = torch.Tensor([math.pi]).to('cuda')
        Oaz0, Taz0 = Oaz0*PI/180, Taz0*PI/180
        Oel0, Tel0 = Oel0*PI/180, Tel0*PI/180
        Oaz1, Taz1 = Oaz1*PI/180, Taz1*PI/180
        Oel1, Tel1 = Oel1*PI/180, Tel1*PI/180
        Oaz2, Taz2 = Oaz2*PI/180, Taz2*PI/180
        Oel2, Tel2 = Oel2*PI/180, Tel2*PI/180

        dist0 = torch.sin(Oel0) * torch.sin(Tel0) + torch.cos(Oel0) * torch.cos(Tel0) * torch.cos(torch.abs(Oaz0 - Taz0))
        dist1 = torch.sin(Oel1) * torch.sin(Tel1) + torch.cos(Oel1) * torch.cos(Tel1) * torch.cos(torch.abs(Oaz1 - Taz1))
        dist2 = torch.sin(Oel2) * torch.sin(Tel2) + torch.cos(Oel2) * torch.cos(Tel2) * torch.cos(torch.abs(Oaz2 - Taz2))
        # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
        dist0 = torch.clamp(dist0, -1+self.epsilon, 1-self.epsilon)
        dist1 = torch.clamp(dist1, -1+self.epsilon, 1-self.epsilon)
        dist2 = torch.clamp(dist2, -1+self.epsilon, 1-self.epsilon)
        dist0 = torch.acos(dist0) * 180 / torch.Tensor([math.pi]).to('cuda')
        dist1 = torch.acos(dist1) * 180 / torch.Tensor([math.pi]).to('cuda')
        dist2 = torch.acos(dist2) * 180 / torch.Tensor([math.pi]).to('cuda')
        return torch.cat((dist0, dist1, dist2), dim=2)

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:,:]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        mask_A0A0A0 = torch.cat((target[:, :, 0, 0:1, :], target[:, :, 0, 0:1, :], target[:, :, 0, 0:1, :]), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        mask_B0B0B1 = torch.cat((target[:, :, 1, 0:1, :], target[:, :, 1, 0:1, :], target[:, :, 2, 0:1, :]), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        mask_B0B1B0 = torch.cat((target[:, :, 1, 0:1, :], target[:, :, 2, 0:1, :], target[:, :, 1, 0:1, :]), 2)
        mask_B0B1B1 = torch.cat((target[:, :, 1, 0:1, :], target[:, :, 2, 0:1, :], target[:, :, 2, 0:1, :]), 2)
        mask_B1B0B0 = torch.cat((target[:, :, 2, 0:1, :], target[:, :, 1, 0:1, :], target[:, :, 1, 0:1, :]), 2)
        mask_B1B0B1 = torch.cat((target[:, :, 2, 0:1, :], target[:, :, 1, 0:1, :], target[:, :, 2, 0:1, :]), 2)
        mask_B1B1B0 = torch.cat((target[:, :, 2, 0:1, :], target[:, :, 2, 0:1, :], target[:, :, 1, 0:1, :]), 2)
        mask_C0C1C2 = torch.cat((target[:, :, 3, 0:1, :], target[:, :, 4, 0:1, :], target[:, :, 5, 0:1, :]), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        mask_C0C2C1 = torch.cat((target[:, :, 3, 0:1, :], target[:, :, 5, 0:1, :], target[:, :, 4, 0:1, :]), 2)
        mask_C1C0C2 = torch.cat((target[:, :, 4, 0:1, :], target[:, :, 3, 0:1, :], target[:, :, 5, 0:1, :]), 2)
        mask_C1C2C0 = torch.cat((target[:, :, 4, 0:1, :], target[:, :, 5, 0:1, :], target[:, :, 3, 0:1, :]), 2)
        mask_C2C0C1 = torch.cat((target[:, :, 5, 0:1, :], target[:, :, 3, 0:1, :], target[:, :, 4, 0:1, :]), 2)
        mask_C2C1C0 = torch.cat((target[:, :, 5, 0:1, :], target[:, :, 4, 0:1, :], target[:, :, 3, 0:1, :]), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2  # ??????????????
        pad4B = target_A0A0A0 + target_C0C1C2  # ??????????????
        pad4C = target_A0A0A0 + target_B0B0B1  # ??????????????
        loss_0 = self._each_calc(output,target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        #         output = get_triplets(output)

        def cart2polar_triplets(tensor):
            x0 = tensor[:, :, [0], :]
            y0 = tensor[:, :, [1], :]
            z0 = tensor[:, :, [2], :]
            x1 = tensor[:, :, [3], :]
            y1 = tensor[:, :, [4], :]
            z1 = tensor[:, :, [5], :]
            x2 = tensor[:, :, [6], :]
            y2 = tensor[:, :, [7], :]
            z2 = tensor[:, :, [8], :]

            azimuth0 = torch.atan2(y0, x0) * 180 / torch.Tensor([math.pi]).to('cuda')
            azimuth1 = torch.atan2(y1, x1) * 180 / torch.Tensor([math.pi]).to('cuda')
            azimuth2 = torch.atan2(y2, x2) * 180 / torch.Tensor([math.pi]).to('cuda')
            elevation0 = torch.atan2(z0, torch.sqrt(x0 ** 2 + y0 ** 2)) * 180 / torch.Tensor([math.pi]).to('cuda')
            elevation1 = torch.atan2(z1, torch.sqrt(x1 ** 2 + y1 ** 2)) * 180 / torch.Tensor([math.pi]).to('cuda')
            elevation2 = torch.atan2(z2, torch.sqrt(x2 ** 2 + y2 ** 2)) * 180 / torch.Tensor([math.pi]).to('cuda')
            return torch.cat((azimuth0, elevation0, azimuth1, elevation1, azimuth2, elevation2), dim=2)

            # convert target and output to polar

        output_polar = cart2polar_triplets(output)
        target_A0A0A0_polar = cart2polar_triplets(target_A0A0A0 + pad4A)
        target_B0B0B1_polar = cart2polar_triplets(target_B0B0B1 + pad4B)
        target_B0B1B0_polar = cart2polar_triplets(target_B0B1B0 + pad4B)
        target_B0B1B1_polar = cart2polar_triplets(target_B0B1B1 + pad4B)
        target_B1B0B0_polar = cart2polar_triplets(target_B1B0B0 + pad4B)
        target_B1B0B1_polar = cart2polar_triplets(target_B1B0B1 + pad4B)
        target_B1B1B0_polar = cart2polar_triplets(target_B1B1B0 + pad4B)
        target_C0C1C2_polar = cart2polar_triplets(target_C0C1C2 + pad4C)
        target_C0C2C1_polar = cart2polar_triplets(target_C0C2C1 + pad4C)
        target_C1C0C2_polar = cart2polar_triplets(target_C1C0C2 + pad4C)
        target_C1C2C0_polar = cart2polar_triplets(target_C1C2C0 + pad4C)
        target_C2C0C1_polar = cart2polar_triplets(target_C2C0C1 + pad4C)
        target_C2C1C0_polar = cart2polar_triplets(target_C2C1C0 + pad4C)

        #         target_A0A0A0_polar = cart2polar_triplets(target_A0A0A0 + pad4A)
        # calculate the angular distance between polar target and output

        loss_0_ang_dist = self._each_calc_angular(output_polar, target_A0A0A0_polar)
        loss_1_ang_dist = self._each_calc_angular(output_polar, target_B0B0B1_polar)
        loss_2_ang_dist = self._each_calc_angular(output_polar, target_B0B1B0_polar)
        loss_3_ang_dist = self._each_calc_angular(output_polar, target_B0B1B1_polar)
        loss_4_ang_dist = self._each_calc_angular(output_polar, target_B1B0B0_polar)
        loss_5_ang_dist = self._each_calc_angular(output_polar, target_B1B0B1_polar)
        loss_6_ang_dist = self._each_calc_angular(output_polar, target_B1B1B0_polar)
        loss_7_ang_dist = self._each_calc_angular(output_polar, target_C0C1C2_polar)
        loss_8_ang_dist = self._each_calc_angular(output_polar, target_C0C2C1_polar)
        loss_9_ang_dist = self._each_calc_angular(output_polar, target_C1C0C2_polar)
        loss_10_ang_dist = self._each_calc_angular(output_polar, target_C1C2C0_polar)
        loss_11_ang_dist = self._each_calc_angular(output_polar, target_C2C0C1_polar)
        loss_12_ang_dist = self._each_calc_angular(output_polar, target_C2C1C0_polar)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        thr = self.threshold

        loss_thr_num = (torch.sum(loss_0_ang_dist * mask_A0A0A0 * torch.unsqueeze((loss_min == 0), 2)) +
                        torch.sum(loss_1_ang_dist * mask_B0B0B1 * torch.unsqueeze((loss_min == 1), 2)) +
                        torch.sum(loss_2_ang_dist * mask_B0B1B0 * torch.unsqueeze((loss_min == 2), 2)) +
                        torch.sum(loss_3_ang_dist * mask_B0B1B1 * torch.unsqueeze((loss_min == 3), 2)) +
                        torch.sum(loss_4_ang_dist * mask_B1B0B0 * torch.unsqueeze((loss_min == 4), 2)) +
                        torch.sum(loss_5_ang_dist * mask_B1B0B1 * torch.unsqueeze((loss_min == 5), 2)) +
                        torch.sum(loss_6_ang_dist * mask_B1B1B0 * torch.unsqueeze((loss_min == 6), 2)) +
                        torch.sum(loss_7_ang_dist * mask_C0C1C2 * torch.unsqueeze((loss_min == 7), 2)) +
                        torch.sum(loss_8_ang_dist * mask_C0C2C1 * torch.unsqueeze((loss_min == 8), 2)) +
                        torch.sum(loss_9_ang_dist * mask_C1C0C2 * torch.unsqueeze((loss_min == 9), 2)) +
                        torch.sum(loss_10_ang_dist * mask_C1C2C0 * torch.unsqueeze((loss_min == 10), 2)) +
                        torch.sum(loss_11_ang_dist * mask_C2C0C1 * torch.unsqueeze((loss_min == 11), 2)) +
                        torch.sum(loss_12_ang_dist * mask_C2C1C0 * torch.unsqueeze((loss_min == 12), 2))
                        )

        loss_thr_den = (torch.sum(mask_A0A0A0) +
                        torch.sum(mask_B0B0B1) +
                        torch.sum(mask_B0B1B0) +
                        torch.sum(mask_B0B1B1) +
                        torch.sum(mask_B1B0B0) +
                        torch.sum(mask_B1B0B1) +
                        torch.sum(mask_B1B1B0) +
                        torch.sum(mask_C0C1C2) +
                        torch.sum(mask_C0C2C1) +
                        torch.sum(mask_C1C0C2) +
                        torch.sum(mask_C1C2C0) +
                        torch.sum(mask_C2C0C1) +
                        torch.sum(mask_C2C1C0))

        loss_thr = torch.clamp(loss_thr_num / loss_thr_den, min=thr) - thr

        return loss + (loss_thr*self.thr_loss_scale)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x

class ConvBlock_unet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):

        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x

class CRNN(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.conv_block_list = torch.nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = params['nb_cnn2d_filt'] * int( np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )


    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        '''(batch_size, feature_maps, time_steps, mel_bins)'''
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            # out - batch x hidden x seq
            x = torch.tanh(x)

        x_rnn = x
        for fnn_cnt in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        return doa
