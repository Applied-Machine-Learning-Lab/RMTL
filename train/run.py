import os
import time
from typing import List
import pickle
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from IPython.display import clear_output
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset.rtrl import RetailRocketRLDataset
from slmodels.esmm import ESMMModel
from slmodels.singletask import SingleTaskModel
from slmodels.ple import PLEModel
from slmodels.mmoe import MMoEModel
from slmodels.sharedbottom import SharedBottomModel
from slmodels.aitm import AITMModel
from slmodels.omoe import OMoEModel

"""
SL run begin here
"""

def get_dataset(name, path):
    if 'rt' in name:  # 当前只支持一个数据集
        return RetailRocketRLDataset(path)
    elif 'kuai' in name:
        return KuaiRLDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'singletask':
        print("Model: SingleTask")
        return SingleTaskModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'esmm':
        print("Model: ESMM")
        return ESMMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2), specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


def sltrain(model, optimizer, data_loader, criterion, device, polisher=None, log_interval=100):
    model.train()
    total_loss = 0
    epoch_loss = []
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (_, categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        if polisher is not None:
            loss = polisher.polish_loss(categorical_fields, numerical_fields, labels, y)
        else:
            loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
            loss = 0
            for item in loss_list:
                loss += item
            loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoch_loss.append(loss.item())
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    return np.mean(epoch_loss)


def sltest(model, data_loader, task_num, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    sessions = []
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for session_id, categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            sessions.extend(session_id.tolist())
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())

    # compute session logloss
    loss_dict["session"] = sessions
    loss_df = pd.DataFrame(loss_dict)
    s_avg = loss_df.groupby(["session"]).mean().mean().tolist()
    return auc_results, loss_results, s_avg, loss_df

def slpred(model, data_loader, task_num, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        predicts_dict[i]= list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                predicts_dict[i].extend(y[i].tolist())
    return predicts_dict


"""
RL run begin here
"""
def one_iter_offline(agent):
    agent.is_test = False
    critic_loss, critic_loss2, actor_loss1, actor_loss2 = agent.update()
    return critic_loss, critic_loss2, actor_loss1, actor_loss2


def train_and_test_offline(environment: gym.Env, agent, epoch, init_episode, save_dir, memory_path):
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # warm replay memory
    if os.path.isfile(memory_path):
        with open(memory_path, "rb") as f:
            agent.memory = pickle.load(f)
    else:
        for _ in range(init_episode):
            state = environment.reset()
            while True:
                action = environment.action_space.sample()
                nstate, reward, done, label = environment.step(action)
                transition = dict(
                    state=state.reshape(-1),
                    action=action.reshape(-1),
                    nstate=nstate.reshape(-1),
                    reward=reward,
                    done=done,
                    label=label
                )
                # print(transition)
                agent.memory.store(**transition)
                state = nstate
                if done:
                    break
        with open(memory_path, "wb") as f:
            pickle.dump(agent.memory, f)

    print("memory size:", agent.memory.size)
    print("epoch | score | q_loss | ac_loss | a_loss | time")
    critic_lossls1, critic_lossls2, actor_lossls1, actor_lossls2, scores = [], [], [], [], []
    # best_auc = 0
    # early_stopper = EarlyStopper(save_path=save_dir,num_trials=2)
    start = time.time()
    for i in range(0, epoch):
        # if i >=50 and (i%5 == 0 or i == epoch - 1):
        #     print(i, agent.total_step, "testing performance")
        #     test_auc = np.sum(test(test_env, agent, save_dir))
        #     if test_auc > best_auc:
        #         best_auc = test_auc
        #         agent.save_or_load_agent(save_dir, if_save=True)
        #     res_df = pd.DataFrame(np.array([critic_lossls1, critic_lossls2, actor_lossls1, actor_lossls2]).T,
        #                           columns=["c1", "c2", "a1", "a2"])
        #     # TODO: tensorboarc sum
        #     res_df.to_csv(save_dir + "/losses.csv")

        critic_loss, critic_loss2, actor_loss1, actor_loss2 = one_iter_offline(agent)

        with torch.no_grad():
            critic_lossls1.append(critic_loss)
            critic_lossls2.append(critic_loss2)
            actor_lossls1.append(actor_loss1)
            actor_lossls2.append(actor_loss2)

        scores.append((0,0))
        critic_lossls = [critic_lossls1[j]+critic_lossls2[j] for j in range(i)]
        # print the train and test performance, can plot here
        if i % 2 == 0 or i == epoch - 1:
            agent.save_or_load_agent(save_dir, if_save=True)
            end = time.time()
            with torch.no_grad():  # draw without considering separate reward
                print(i, np.mean(np.array(scores), axis=0), np.mean(critic_lossls), np.mean(actor_lossls1),
                      np.mean(actor_lossls2), end - start)
                plot_loss(i, np.sum(np.array(scores), axis=1), critic_lossls, actor_lossls1, actor_lossls2)
            start = time.time()

    environment.close()


def test(environment, agent, save_dir):
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    agent.is_test = True
    agent.actor.eval()
    pos_score = []

    preds = []
    labels = []
    for _ in range(environment.datalen):
        # for i in range(1000):
        state = environment.reset()
        score = np.zeros(2)
        nsteps = 0
        # pred = []
        while True:
            action = agent.select_action(state)
            nstate, reward, done, label = environment.step(action)
            score += reward

            # store the CTR/CVR and true label
            # 注意：此处为separate learning
            preds.append([action[0], action[1]])
            labels.append(label)
            state = nstate
            nsteps += 1
            if done:
                break
        pos_score.append(score / nsteps)
    preds = np.array(preds)
    labels = np.array(labels)
    res = pd.DataFrame(np.c_[preds, labels])
    res.to_csv(f"{save_dir}/RLpreds.csv", index=False)
    test_loss = [F.binary_cross_entropy(torch.tensor(labels[:, j]), torch.tensor(preds[:, j])).item() for j in range(2)]
    test_auc = [roc_auc_score(labels[:, j], preds[:, j]) for j in range(2)]
    print("score:{}, test logloss:{}, test auc:{}".format(np.mean(np.array(pos_score), axis=0), test_loss, test_auc))
    # environment.close()
    return test_auc


def plot_loss(
        frame_idx: int,
        scores: List[float],
        critic_losses: List[float],
        actor_losses1: List[float],
        actor_losses2: List[float]
):
    """Plot the training progresses."""

    def subplot(loc: int, title: str, values: List[float]):
        plt.subplot(loc)
        plt.title(title)
        plt.plot(values)

    subplot_params = [
        (221, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
        (222, "critic_loss", critic_losses),
        (223, "ac_loss", actor_losses1),
        (224, "a_loss", actor_losses2),
    ]

    clear_output(True)
    plt.figure(figsize=(30, 5))
    for loc, title, values in subplot_params:
        subplot(loc, title, values)

    plt.show()
