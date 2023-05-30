import os.path

import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
import torch
torch.cuda.current_device()
import torch.nn.functional as F
import gym
from layers.critic import CriticNeg
from slmodels.esmm import ESMMModel
from slmodels.mmoe import MMoEModel


def get_optim_param(optim):  # optim = torch.optim.Adam(network_param, learning_rate)
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


class Catemapper:
    def __init__(self, threshold):
        self.threshold = threshold
        self.catemap = defaultdict(dict)
        self.field_dims = None

    def make_mapper(self, fea_path, columns, filter_cols):
        df = pd.read_csv(fea_path, usecols=columns)
        df.drop_duplicates(inplace=True)
        df.fillna(0, inplace=True)

        length_df = len(df)
        dims = []
        for col in columns:
            df[col].value_counts()
            self.catemap[col] = dict()
            count = 0
            idx = 0
            if col in filter_cols:
                for i, v in df[col].value_counts().to_dict().items():
                    count += v
                    ad = 1
                    if count / length_df > self.threshold:
                        ad = 0
                    self.catemap[col][i] = idx
                    idx += ad
                dims.append(idx)
            else:
                for i, v in df[col].value_counts().to_dict().items():
                    self.catemap[col][i] = idx
                    idx += 1
                dims.append(idx - 1)

        self.field_dims = np.array(dims).astype(np.int64) + 1
        for col in self.catemap:
            mapper = self.catemap[col]
            origin, processed = len(set(mapper.keys())), len(set(mapper.values()))
            print("{}: {} to {}".format(col, origin, processed))

    def save_mapper(self, save_path):
        with open(save_path + "/catemap.pkl", 'wb') as f:
            pickle.dump(self.catemap, f)

    def load_mapper(self, save_path):
        with open(save_path + "/catemap.pkl", 'rb') as f:
            self.catemap = pickle.load(f)
        dims = []
        for i in self.catemap:
            dims.append(list(self.catemap[i].values())[-1])
        self.field_dims = np.array(dims).astype(np.int64) + 1

    def map_rt(self, dataset):
        tmp_df = pd.DataFrame(dataset.categorical_data, columns=dataset.cate_cols)
        for col in self.catemap:
            tmp_df[col] = tmp_df[col].apply(lambda x: self.catemap[col][x] if x in self.catemap[col] else 0)
        dataset.categorical_data = tmp_df.values.astype(np.int)
        dataset.field_dims = self.field_dims

    def map(self, df):
        for col in self.catemap:
            df[col] = df[col].apply(
                lambda x: self.catemap[col][x] if x in self.catemap[col] else 0)


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class ConvergeStopper(object):
    def __init__(self, save_path, num_trials=2, eps=3e-3):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.last_loss = 1e3
        # self.accuracy = 0
        self.eps = eps
        self.save_path = save_path

    def is_continuable(self, agent, loss):
        if np.abs(np.mean(loss - self.last_loss)) > self.eps:
            self.last_loss = loss
            self.trial_counter = 0
            # if accu > self.accuracy:
            #     self.accuracy = accu
            #     torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            # self.last_loss = loss  # maybe not useful
            self.trial_counter += 1
            return True
        else:
            return False


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, 0.0, 1.0)

        return action


class RlLossPolisher:
    def __init__(self, env, model_name, lambda_=0.5):
        # tuning param
        self.lambda_ = lambda_

        # dynamic path
        self.rl_path = f"./chkpt/RL/res_TD3BC_{model_name}"
        if not os.path.isdir(self.rl_path):
            raise FileNotFoundError
        # fixed params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.categorical_field_dims = env.field_dims
        self.num_dim = 1
        self.task_num = env.action_space.shape[0]
        self.pretrain_path = self.rl_path+f"/rt_{model_name}.pt"
        self.embed_dim = 128
        self.bottom_mlp_dims = (512, 256)
        self.tower_mlp_dims = (128, 64)
        self.drop_out = 0.2

        # define the network
        # self.pretain_actor = ESMMModel(self.categorical_field_dims, self.num_dim, self.embed_dim, self.bottom_mlp_dims,
        #                                self.tower_mlp_dims,
        #                                self.task_num, self.drop_out).to(self.device)
        # self.pretain_actor.load_state_dict(torch.load(self.pretrain_path))
        # self.pretain_actor.eval()

        # TODO: polish to code, use a general critic network to contain multiple task;
        #  以及是否丧失了MDP特性，critic是R^t_{\pi}，此处无pi；暂时作为logging policy对待
        self.critic1 = CriticNeg(self.categorical_field_dims, self.num_dim, self.embed_dim, self.bottom_mlp_dims,
                              self.tower_mlp_dims, self.drop_out).to(self.device)

        self.critic2 = CriticNeg(self.categorical_field_dims, self.num_dim, self.embed_dim, self.bottom_mlp_dims,
                              self.tower_mlp_dims, self.drop_out).to(self.device)

        state_dict1 = torch.load(self.rl_path + "/critic1.pth", map_location=lambda storage, loc: storage)
        self.critic1.load_state_dict(state_dict1)
        state_dict2 = torch.load(self.rl_path + "/critic2.pth", map_location=lambda storage, loc: storage)
        self.critic2.load_state_dict(state_dict2)

    def polish_loss(self, categorical_fields, numerical_fields, labels, y):
        # default two task here
        slloss = [torch.nn.BCELoss(reduction='none')(y[i],labels[:,i]) for i in range(2)]

        q_weight = [self.critic1(categorical_fields, numerical_fields, torch.unsqueeze(y[0], 1)),
                    self.critic2(categorical_fields, numerical_fields, torch.unsqueeze(y[1], 1))]

        # method 1
        # loss_list = [(1 - self.lambda_ * labels[:, i] * q_weight[i].detach()) *
        #              slloss[i] for i in range(2)]

        # method 2
        # loss_list = [0.5 * slloss[i] for i in range(2)]

        loss_list = [(1 - self.lambda_ * q_weight[i].detach()) *
                     slloss[i] for i in range(2)]

        # method 3
        #loss_list = [(1-self.lambda_ * q_weight[i].detach()) *
        #       slloss[i] for i in range(2)]

        # method 4
        # loss_list = [(0-q_weight[i].detach()) * slloss[i] for i in range(2)]

        loss = 0
        for item in loss_list:
            loss += torch.mean(item)
        loss /= len(loss_list)


        # method 4 plus BC
        # ref_loss = 0  # no need to give mode generalization
        # param_count = 0
        # for param, value in self.slmodel.named_parameters():
        #    param_count += 1
        #     ref_loss += F.mse_loss(value, self.pretain_actor.state_dict()[param])
        #     ref_loss /= param_count

        #    lambda_ = self.actor_reg / torch.mean(-q1_loss_weight - q2_loss_weight).abs().detach()
        #    ac_loss = lambda_ * ac_loss + ref_loss
        #    loss += self.reg_rate * ref_loss
        return loss


if __name__ == '__main__':
    features_path = "../dataset/item_feadf.csv"
    columns = ['785', '591', '814', 'available', 'categoryid', '364', '776']
    filter_cols = ['776', '364']
    cm = Catemapper(0.2)
    cm.make_mapper(features_path, columns, filter_cols)
    cm.save_mapper("./chkpt")
    cm.load_mapper("./chkpt")
