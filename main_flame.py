import os
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import trainer.accuracy as accuracy
import utils.options as options
import utils.posion_info as posion_info
import utils.tools as tools
from utils.sampling import model_and_data
from trainer.client import create_model
from trainer.client import local_train
from trainer.filter import *
from utils.util import *


def train():
    args = options.agg_flame()
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    INFO_LOG.logger.info(args)
    aa = args['num_of_attack']
    all_trigger_lst, trigger_of_acs = posion_info.creat_trigger(args, aa)
    train_dataset, test_dataset, train_iter_list, test_iter_list, poison_test_iter = model_and_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args['batchsize'], shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batchsize'], shuffle=True, num_workers=4)

    global_net = create_model(args).to(dev)
    # global_net = torch.nn.DataParallel(global_net) 

    local_net = copy.deepcopy(global_net)

    global_weight = tools.shape_to_1dim(global_net, single=True)

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1)) 
    if num_in_comm < 10:
        num_in_comm = 10
    # num_in_comm = 5

    print("_________________START TRAIN________________________")
    main_acc_local, main_acc_global = [], []
    backdoor_acc = [[] for _ in range(aa + 1)]
    true_positive_rate_list = []
    true_negative_rate_list = []
    train_acc_global = []
    select_t_list = []
    admit_t_list = []
    # epoch_under_attack = list(np.random.choice(range(args['num_comm']), int(args['attack_epoch_num']), replace=False))
    epoch_under_attack = list(range(args["attack_epoch_in"], args["num_comm"]))
    acs = list(range(args['num_of_attack']))

    for epoch in tqdm(range(args['num_comm']), desc='Training: ', colour='green'):

        if_ATTACK = True if epoch in epoch_under_attack else False
        benigns = list(
            np.random.choice(
                range(args['num_of_attack'], args['num_of_clients']),
                num_in_comm - args['num_of_attack'],
                replace=False
            )
        )
        local_choose = acs + benigns

        local_choose.sort()

        # 开始训练
        local_update_lts = []
        global_weight = tools.shape_to_1dim(global_net, single=True)
        for i in range(num_in_comm):
            local = local_choose[i]
            tools.hand_out(global_net, local_net)
            net, train_iter = local_net, train_iter_list[local]

            if local in acs and if_ATTACK:  # 毒化
                # INFO_LOG.logger.info("Attack")
                poison_info = trigger_of_acs[local]
                opt = torch.optim.SGD(
                    net.parameters(),
                    lr=poison_info['poison_lr'],
                    momentum=0.9,
                    weight_decay=args['weight_decay']
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=[0.2 * args['local_poison_epoch'], 0.8 * args['local_poison_epoch']],
                    gamma=0.1
                )
                local_train(
                    net, train_iter, opt, device=dev,
                    num_epochs=args['local_poison_epoch'],
                    poison_info=poison_info,
                    scheduler=scheduler,
                    dataset_name=args['dataset']
                )
            else:
                opt = torch.optim.SGD(
                    net.parameters(),
                    lr=args['learning_rate'], momentum=0.9,
                    weight_decay=args['weight_decay']
                )
                local_train(
                    net, train_iter, opt, device=dev,
                    num_epochs=args['epoch'],
                    dataset_name=args['dataset']
                )
            # local_update_lts.shape = (10, 2797610)
            local_update_lts.append(tools.shape_to_1dim(net, single=True))  # update full model

        if not if_ATTACK:
            admitted_index = local_choose
            # local_net_params.shape = (10, 62)
            local_net_params = tools.shape_back(local_update_lts, local_net)

            sum_params = np.array(local_net_params).sum(axis=0)
            for g_p, l_sum_p in zip(global_net.parameters(), sum_params):
                l_sum_p /= len(admitted_index)  # FedAvg
                l_sum_p = l_sum_p.to(dev)
                g_p.data = l_sum_p.data
        else:
            with torch.no_grad():
                print("clients index: ", local_choose)
 
                admitted_index_in_wlst = model_filtering_layer(local_update_lts, args)
                admitted_index = []
                for i in admitted_index_in_wlst:
                    admitted_index.append(local_choose[i])
                admitted_index.sort()

                if args['clip']:
                    local_update_lts, St = adaptive_clipping(global_weight, local_update_lts, admitted_index_in_wlst)


                benign_w = []
                for i in admitted_index_in_wlst:
                    benign_w.append(local_update_lts[i])
                local_net_params = tools.shape_back(benign_w, local_net)

                sum_params = np.array(local_net_params).sum(axis=0)
                # sum_params = local_net_params[0]
                # for i in range(1, len(local_net_params)):
                #     for s, l in zip(sum_params, local_net_params[i]):
                #         s.data += l.data

                for g_p, l_sum_p in zip(global_net.parameters(), sum_params):
                    l_sum_p /= len(admitted_index)  # FedAvg
                    l_sum_p = l_sum_p.to(dev)
                    g_p.data = l_sum_p.data


                if args['noise']:
                    alpha = args['λ'] * St
                    adaptive_noising(global_net, alpha)

        # if not if_ATTACK:
        #     true_negative_rate, true_positive_rate = 1.0, 1.0
        # else:
        true_positive_rate, select_t_1,admit_t_1 = get_true_positive_rate(admitted_index, acs)
        true_negative_rate, select_t_2,admit_t_2 = get_true_negative_rate(admitted_index, acs)

        true_positive_rate_list.append(true_positive_rate)
        true_negative_rate_list.append(true_negative_rate)
        select_t_list.append(select_t_1)
        admit_t_list.append(admit_t_1)
        INFO_LOG.logger.info(f'true_positive_rate: {true_positive_rate_list}')
        INFO_LOG.logger.info(f'true_negative_rate: {true_negative_rate_list}')

        train_acc = accuracy.evaluate_accuracy(
            train_dataloader,
            global_net,
            dataset_name=args['dataset']
        )
        train_acc_global.append(train_acc)

        test_acc = accuracy.evaluate_accuracy(
            test_dataloader,
            global_net,
            dataset_name=args['dataset']
        )
        main_acc_global.append(test_acc)

        for i in range(aa + 1):
            acc = 0.0
            if if_ATTACK:
                acc = accuracy.evaluate_accuracy(
                    poison_test_iter,
                    global_net,
                    poison_info=all_trigger_lst[i],
                    dataset_name=args['dataset']
                )
            backdoor_acc[i].append(acc)

        INFO_LOG.logger.info(f'train_acc: {train_acc_global}')
        INFO_LOG.logger.info(f'main_take_acc: {main_acc_global}')
        INFO_LOG.logger.info(f'global_trigger_acc: {backdoor_acc[0]}')

    if args['IID'] == False: 
        save_path = f"tmp/IID/{args['dataset']}/{args['IID_rate']}-{args['defend']}-{args['dataset']}-poison_rate-{trigger_of_acs[0]['poison_rate']}-{getNowTime()}-{args['learning_rate']}.pth"
    else:
        save_path = f"tmp/client-{args['num_of_clients']}/fedward/{args['dataset']}/{args['IID_rate']}-{args['defend']}-{args['dataset']}-poison_rate-{trigger_of_acs[0]['poison_rate']}-{getNowTime()}-{args['learning_rate']}.pth"
    INFO_LOG.logger.info(f'save result in file:\n {save_path}')

    torch.save(
        [
            train_acc_global,
            main_acc_global,
            backdoor_acc,
            true_positive_rate_list,
            true_negative_rate_list,
            select_t_list,
            admit_t_list
        ],
        save_path
    )


if __name__ == '__main__':
    train()