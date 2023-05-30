import torch
import random
import tqdm
from sklearn.metrics import roc_auc_score
# pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
# from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd

# 导入数据集
from dataset.rtrl import RetailRocketRLDataset

# 导入MTL模型
# from layers.esmm import ESMMModel
from slmodels.esmm import ESMMModel
from slmodels.singletask import SingleTaskModel
from slmodels.ple import PLEModel
from slmodels.mmoe import MMoEModel
from slmodels.sharedbottom import SharedBottomModel
from slmodels.aitm import AITMModel
from slmodels.omoe import OMoEModel

# 导入强化学习环境
from train.utils import Catemapper,EarlyStopper,ActionNormalizer,RlLossPolisher
from env import MTEnv
from train.Arguments import Arguments
from train.run import get_model, get_dataset, sltrain as train, sltest as test, slpred as pred

def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         feature_map_rate,
         batch_size,
         embed_dim,
         weight_decay,
         polish_lambda,
         device,
         save_dir):
    device = torch.device(device)
    # 装载数据集
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv')
    val_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv')
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/val.csv')

    catemap = Catemapper(threshold=feature_map_rate)
    # 没有与训练的类别筛选就使用下面3行
    # catemap.make_mapper(os.path.join(dataset_path, dataset_name)+'/item_feadf.csv',
    #                     train_dataset.cate_cols,train_dataset.filter_cols)
    # catemap.save_mapper(save_dir)
    catemap.load_mapper("./pretrain")  # ABSOLUTE path
    catemap.map_rt(train_dataset)
    catemap.map_rt(val_dataset)
    catemap.map_rt(test_dataset)
    print("categorical data map successfully!")

    # balance sampling，非平衡采样，没什么效果
    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,sampler=ImbalancedDatasetSampler(train_dataset))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # 模型训练，二轮训练得删除下面环境注释
    """
    # define test environment
    hyparams = Arguments()
    hyparams.test_rows = 50000
    test_env = ActionNormalizer(MTEnv("./dataset/rt/test_set.csv", hyparams.features_path, hyparams.map_path,
                                      reward_type=hyparams.reward_type, nrows=hyparams.test_rows, is_test=True))
    test_env.getMDP()
    """
    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    print("field_dims:",field_dims)
    model = get_model(model_name, field_dims, numerical_num, task_num, expert_num, embed_dim).to(device)
    print(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_dir = f'{save_dir}/{dataset_name}_{model_name}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/{dataset_name}_{model_name}_{polish_lambda}.pt'
    # save_path = "./pretrain/rt_esmm.pt"
    # Contribution: use loss polisher
    # model.load_state_dict(torch.load(f'{save_dir}/{dataset_name}_ple_ple0.0.pt'))

    if polish_lambda != 0:
        hyparams = Arguments()
        hyparams.test_rows = 500
        test_env = ActionNormalizer(MTEnv(hyparams.test_path, hyparams.features_path, hyparams.map_path,
                                          reward_type=hyparams.reward_type, nrows=hyparams.test_rows, is_test=True))
        test_env.getMDP()
        polisher = RlLossPolisher(test_env, model_name, lambda_=polish_lambda)
        model.load_state_dict(torch.load(f'{save_dir}/{dataset_name}_{model_name}_0.0.pt'))
    else:
        polisher = None
    # polisher = RlLossPolisher(test_env, "esmm", lambda_=polish_lambda)  # test transibility
    early_stopper = EarlyStopper(num_trials=2, save_path=save_path)
    for epoch_i in range(epoch):
        train_loss = train(model, optimizer, train_data_loader, criterion, device, polisher)
        auc, loss, _, _ = test(model, val_data_loader, task_num, device)
        # auc, loss = env_test(test_env, model, save_dir, device)
        print('epoch:', epoch_i,'train loss:',train_loss, 'test: auc:', auc)
        # print('epoch:', epoch_i, 'train loss:', train_loss)
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))

        # auc stopper
        if not early_stopper.is_continuable(model, np.array(auc).mean()):
            print(f'test: best auc: {early_stopper.best_accuracy}')
            break


    # save_path = f'{save_dir}/{dataset_name}_mmoe_0.0.pt' # test directly
    model.load_state_dict(torch.load(save_path))
    auc, loss, sloss, loss_df = test(model, test_data_loader, task_num, device)
    print("session_loss",sloss)
    # env test
    # auc, loss = env_test(test_env, model, save_dir, device)
    f = open(save_dir + '/{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print('task {}, AUC {}, Log-loss {}, session logloss {}'.format(i, auc[i], loss[i], sloss[i]))
        f.write('task {}, AUC {}, Log-loss {}, session logloss {}\n'.format(i, auc[i], loss[i], sloss[i]))
    print(loss_df.groupby(["session"]).mean())
    print('\n')
    f.write('\n')
    f.close()


    # output the predictions
    # data_loader1 = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    # data_loader2 = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    # train_pred_df = pd.DataFrame(pred(model, data_loader1, task_num, device))
    # test_pred_df = pd.DataFrame(pred(model, data_loader2, task_num, device))
    # res = pd.concat([train_pred_df,test_pred_df],ignore_index=True)
    # res.to_csv(save_dir+"/res{}.csv".format(model_name),index=False)
    



if __name__ == '__main__':
    import argparse
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='rt', choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US',"rt","rsc"])
    parser.add_argument('--dataset_path', default='./dataset/')
    parser.add_argument('--model_name', default='esmm', choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm','esmm'])
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--polish', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--feature_map_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./chkpt/SL')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.feature_map_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.polish,
         args.device,
         args.save_dir)
