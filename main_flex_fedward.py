import copy
import os
from tqdm import tqdm
import utils.tools as tools
from trainer.filter import *
import trainer.defend as defend 
import utils.options as options
import trainer.config as config
import trainer.accuracy as accuracy
import utils.posion_info as posion_info
from trainer.client import local_train
from trainer.client import create_model
from torch.utils.data import DataLoader
from utils.sampling import model_and_data
from utils.util import nowTime, INFO_LOG, get_true_negative_rate, get_true_positive_rate

def train():
    args = options.agg_des()
    config.setting(args)
    # args = parser.parse_args()
    # args = args.__dict__  

    INFO_LOG.logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("dev", torch.cuda.is_available())


    train_dataset, test_dataset, train_iter_list, test_iter_list, poison_test_iter = model_and_data(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batchsize'], shuffle=True, num_workers=4)
    global_net = create_model(args).to(dev)
    emptry_net = copy.deepcopy(global_net)
    local_net = copy.deepcopy(global_net)

    global_weight = tools.shape_to_1dim(global_net, single=True)
    w_size = len(global_weight)

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))  
    if num_in_comm < 10:
        num_in_comm = 10
    # num_in_comm = 5

    print("_________________START TRAIN________________________")
    main_acc_local, main_acc_global = [], []
    nnum = args['num_of_attack']
    backdoor_acc = [[] for i in range(nnum + 1)]
    true_positive_rate_list = []
    true_negative_rate_list = []
    select_t_list = []
    admit_t_list = []
    all_trigger_lst, trigger_of_acs = posion_info.creat_trigger(args, nnum)
    epoch_under_attack = list(range(args["attack_epoch_in"], args["num_comm"]))

    acs = list(range(args['num_of_attack']))
    for epoch in tqdm(range(args['num_comm']), desc='Training: '):
        if_ATTACK = True if epoch in epoch_under_attack else False
        benigns = list(
            np.random.choice(range(args['num_of_attack'], args['num_of_clients']), num_in_comm - args['num_of_attack'],
                             replace=False))
        local_choose = acs + benigns

        local_choose.sort()

        local_update_lts = []
        alpha_lts = []
        alphas = []
        alpha_local = []
        alphas_layers = []
        
        if args['defend'] == 'fedcc':
            global_net_alphas_layer = get_penultimate_layer(args, global_net)

        global_weight = tools.shape_to_1dim(global_net, single=True)
        for i in range(num_in_comm):
            local = local_choose[i]

            tools.hand_out(global_net, local_net)
            net, train_iter, test_iter = local_net, train_iter_list[local], test_iter_list[local]

            if local in acs and if_ATTACK:  # poison
                poison_info = trigger_of_acs[local]
                # momentum = 0.5
                opt = torch.optim.SGD(net.parameters(), lr=poison_info['poison_lr'], momentum=0.9,
                                      weight_decay=args['weight_decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[0.2 * args['local_poison_epoch'],
                                                                                  0.8 * args['local_poison_epoch']],
                                                                 gamma=0.1)
                local_train(net, train_iter, opt, device=dev, num_epochs=args['local_poison_epoch'],
                            poison_info=poison_info, scheduler=scheduler, dataset_name=args['dataset'])
            else:
                opt = torch.optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9,
                                      weight_decay=args['weight_decay'])
                local_train(net, train_iter, opt, device=dev, num_epochs=args['epoch'],
                            dataset_name=args['dataset'])

            if args['defend'] == 'fedcc':
                alphas_layer = get_penultimate_layer(args, net)
                alphas_layers.append(alphas_layer - global_net_alphas_layer)
                
            local_w = tools.shape_to_1dim(net, single=True)  # update full model
            
            
            alpha_local.append(local_w)
            delta = local_w - global_weight
            local_update_lts.append(np.sign(delta))  # update sign grad
            alphas.append(delta)
            norm1 = np.linalg.norm(delta, ord=1)
            alpha_lts.append(norm1 / w_size)
   
        amgrad = AMGRAD(alphas, emptry_net)
        
        with torch.no_grad():
            if not if_ATTACK:
                admitted_index = local_choose
                local_net_params = tools.shape_back(alpha_local, local_net)
                sum_params = np.array(local_net_params).sum(axis=0)
                for g_p, l_sum_p in zip(global_net.parameters(), sum_params):
                    l_sum_p /= len(admitted_index)  # FedAvg
                    l_sum_p = l_sum_p.to(dev)
                    g_p.data = l_sum_p.data
            else:
                if args['defend'] == 'fedward':
                    INFO_LOG.logger.info(f'clients index:  {local_choose}')
                    admitted_index_in_wlst = auto_optics(amgrad, args)
  
                    # print("admitted_index_in_wlst", admitted_index_in_wlst)
                    admitted_index = []
                    for i in admitted_index_in_wlst:
                        admitted_index.append(local_choose[i])
                    admitted_index.sort()
                    INFO_LOG.logger.info(f'choose model:  {admitted_index}')

                    #autoclipping
                    St = np.median(alpha_lts)
                    benign_w = []
                    for i in admitted_index_in_wlst:
                        benign_w.append(local_update_lts[i] * min(alpha_lts[i], St))

                    avg_grad = sum(benign_w)
                    global_weight += avg_grad
                    global_net = tools.shape_back_to(global_weight, global_net)

    
                elif args['defend'] == 'fedcc':    
                    admitted_index_in_wlst = model_filtering_layer_fedcc(alphas_layers, args)
                    admitted_index = []
                    for i in admitted_index_in_wlst:
                        admitted_index.append(local_choose[i])
                    admitted_index.sort()
                    INFO_LOG.logger.info(f'choose model:  {admitted_index}')
                    benign_w = []
                    for i in admitted_index_in_wlst:
                        benign_w.append(alphas[i])
                    # defend.average(benign_w)
                    benign_w_ = defend.average(benign_w)
                    global_weight += benign_w_
                    global_net = tools.shape_back_to(global_weight, global_net)

                elif args['defend'] == 'crfl':
                    # if args['dataset'] == 'MNIST':
                    #     dynamic_thres = i *0.1+2 # i equal epoch
                    # elif args['dataset'] == 'EMNIST':
                    #     dynamic_thres = i*0.25+4
                    # elif args['dataset'] == 'FASHION':
                    #     dynamic_thres = i*0.25+4
                    # elif args['dataset'] == 'CIFAR':
                    #     dynamic_thres = i*0.25+6
                    # if dynamic_thres < 15:
                    #     dynamic_thres = 15

                    # sigma_param = 0.01
                    # benign_w = defend.average(alphas)
                    # global_weight += benign_w
                    # global_net = tools.shape_back_to(global_weight, global_net)
                    # tools.clip_weight_norm(global_net, dynamic_thres)
                    # for name, param in global_net.state_dict().items():
                    #     param.add_(tools.dp_noise(param, sigma_param))
                    admitted_index =  list(range(len(local_choose)))
                    local_update_lts, St = adaptive_clipping(global_weight, alpha_local, admitted_index)
                    # filter
                    benign_w = []
                    for i in admitted_index:
                        benign_w.append(local_update_lts[i])
                    local_net_params = tools.shape_back(benign_w, local_net)
                 
                    sum_params = np.array(local_net_params).sum(axis=0)
                    for g_p, l_sum_p in zip(global_net.parameters(), sum_params):
                        l_sum_p /= len(admitted_index) 
                        l_sum_p = l_sum_p.to(dev)
                        g_p.data = l_sum_p.data
                    alpha = args['λ'] * St
                    adaptive_noising(global_net, alpha)


                elif args['defend'] == 'median':
                    benign_w = defend.median(alphas)
                    global_weight += benign_w
                    global_net = tools.shape_back_to(global_weight, global_net)


                elif args['defend'] == 'fedavg':
                    benign_w = defend.average(alphas)
                    global_weight += benign_w
                    global_net = tools.shape_back_to(global_weight, global_net)


                elif args['defend'] == 'trimmed_mean':
                    benign_w = defend.trimmed_mean(alphas)
                    global_weight += benign_w
                    global_net = tools.shape_back_to(global_weight, global_net)
             

        print("admitted_index", admitted_index, acs)
        true_positive_rate, select_t_1,admit_t_1 = get_true_positive_rate(admitted_index, acs)
        true_negative_rate, _, _ = get_true_negative_rate(admitted_index, acs)

        true_positive_rate_list.append(true_positive_rate)
        true_negative_rate_list.append(true_negative_rate)
        select_t_list.append(select_t_1)
        admit_t_list.append(admit_t_1)
        # MA main task
        acc = accuracy.evaluate_accuracy(
            test_dataloader,
            global_net,
            dataset_name=args['dataset']
        )
        main_acc_global.append(acc)

     
        for i in range(nnum + 1):
            acc = accuracy.evaluate_accuracy(poison_test_iter, global_net, poison_info=all_trigger_lst[i],
                                             dataset_name=args['dataset'])
            backdoor_acc[i].append(acc)

        INFO_LOG.logger.info(f'true_positive_rate: {true_positive_rate_list}')
        INFO_LOG.logger.info(f'true_negative_rate: {true_negative_rate_list}')
        INFO_LOG.logger.info(f'main_take_acc: {main_acc_global}')
        INFO_LOG.logger.info(f'global_trigger_acc: {backdoor_acc[0]}')

    # save result
    if args['IID'] == False:
        save_path = f"./tmp/client-{args['num_of_clients']}/IID/{args['dataset']}/{args['IID_rate']}-{args['defend']}-{args['dataset']}-{nowTime}-{args['num_of_attack']}-posion_rate({trigger_of_acs[0]['poison_rate']}).pth"
    else:
        save_path = f"./tmp/client-{args['num_of_clients']}/fedward/{args['dataset']}/{args['IID_rate']}-{args['defend']}-{args['dataset']}-{nowTime}-{args['num_of_attack']}-posion_rate({trigger_of_acs[0]['poison_rate']}).pth"
    torch.save(
        [
            main_acc_global,
            backdoor_acc,
            true_positive_rate_list,
            true_negative_rate_list,
            select_t_list,
            admit_t_list
        ],
        save_path
    )
    INFO_LOG.logger.info(f"the result are save in {save_path}")
    INFO_LOG.logger.info(args)

if __name__ == "__main__":
    train()