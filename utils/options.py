import argparse


def agg_des():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-name', '--name', type=str, default="", help='target')
    parser.add_argument('-defend', '--defend', type=str, default="fedward", help='defend')
    parser.add_argument('-g', '--gpu', type=str, default='1', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
    parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')
    parser.add_argument('-iid_rate', '--IID_rate', type=float, default=0.5, help='the rate of iid data')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-NA', '--num_of_attack', type=int, default=4, help='number of attacker')
    parser.add_argument('-target', '--target', type=int, default=2, help='target')
    parser.add_argument('-dataset', '--dataset', type=str, default='CIFAR', help='dataset will be used')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
    parser.add_argument('-AE', '--attack_epoch_in', type=int, default=50, help='number of under attack epoch')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-WD', '--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('-E', '--epoch', type=int, default=3, help='local train epoch')
    parser.add_argument('-LPE', '--local_poison_epoch', type=int, default=6, help='local poison train epoch')
    parser.add_argument('-pr', '--poison_rate', type=float, default=0.3125, help='the poison rate of malicious client')
    parser.add_argument('-sign', '--sign', type=bool, default=True, help='use signSGD to aggregation')
    parser.add_argument('-mn', '--model_name', type=str, default="cnn", help='choose model')
    parser.add_argument('-clip', '--clip', type=bool, default=True)
    parser.add_argument('-noise', '--noise', type=bool, default=True)
    parser.add_argument('-λ', '--λ', type=float, default=0.0001, help='λ')
    parser.add_argument('-v', '--vote', type=bool, default=False, help="Whether to use voting")
    args = parser.parse_args()
    return  args


def agg_flame():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='1', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
    parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')
    parser.add_argument('-iid_rate', '--IID_rate', type=float, default=0.5, help='the rate of iid data')
    parser.add_argument('-mn', '--model_name', type=str, default="cnn", help='choose model')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1,help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-NA', '--num_of_attack', type=int, default=4, help='number of attacker')
    parser.add_argument('-target', '--target', type=int, default=2, help='target')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=220, help='number of communications')
    parser.add_argument('-AE', '--attack_epoch_in', type=int, default=10, help='number of under attack epoch')
    parser.add_argument('-dataset', '--dataset', type=str, default='CIFAR', help='dataset will be used')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-WD', '--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('-E', '--epoch', type=int, default=3, help='local train epoch')
    parser.add_argument('-LPE', '--local_poison_epoch', type=int, default=6, help='local poison train epoch')
    parser.add_argument('-pr', '--poison_rate', type=float, default=0.3125, help='the poison rate of malicious client')
    parser.add_argument('-sign', '--sign', type=bool, default=False, help='use signSGD to aggregation')
    parser.add_argument('-clip', '--clip', type=bool, default=True)
    parser.add_argument('-noise', '--noise', type=bool, default=True)
    parser.add_argument('-λ', '--λ', type=float, default=0.0001, help='λ')
    args = parser.parse_args()
    return  args