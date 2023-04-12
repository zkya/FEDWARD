import logging
import time
import torch
import colorlog


def topk(vec, ret, k):
    """ Return the largest k elements (by magnitude) of vec"""
    # on a gpu, sorting is faster than pytorch's topk method
    # topkIndices = torch.sort(vec**2)[1][-k:]
    # however, torch.topk is more space efficient

    # topk on cuda returns what looks like uninitialized memory if
    # vals has nan values in it
    # saving to a zero-initialized output array instead of using the
    # output of topk appears to solve this problem
    vec = torch.from_numpy(vec) 
    ret = torch.from_numpy(ret)
    topkVals = torch.zeros(k)
    topkIndices = torch.zeros(k).long()
   
    torch.topk(vec ** 2, k, sorted=False, out=(topkVals, topkIndices))
    
    # ret = torch.zeros_like(vec)
    
    if len(vec.size()) == 1:
        ret[topkIndices] = vec[topkIndices]
    elif len(vec.size()) == 2: # n 行一列
        rows = torch.arange(vec.size()[0]).view(-1, 1)
        ret[rows, topkIndices] = vec[rows, topkIndices]

    ret = ret.numpy()

    return ret, topkIndices


class DatasetName():
    TINY = 'TINY'
    MNIST = 'MNIST'
    FASHION = 'FASHION'
    CIFAR = 'CIFAR'
    REDDIT = 'REDDIT'
    IOT_TRAFFIC = 'IOT_TRAFFIC'
    

class LogHandler(object):

    def __init__(self, filename, level=logging.INFO):
        self.logger = logging.getLogger(filename)
        self.log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
            log_colors=self.log_colors_config)

        # 设置日志级别
        self.logger.setLevel(level)
        # 往屏幕上输出
        console_handler = logging.StreamHandler()
        # 输出到文件
        file_handler = logging.FileHandler(filename=filename, mode='a', encoding='utf8')
        file_formatter = logging.Formatter('%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s')
        # 设置屏幕上显示的格式
        console_handler.setFormatter(formatter)
        # 设置写入文件的格式
        file_handler.setFormatter(file_formatter)
        # 把对象加到logger里
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


def getNowTime()->str:
    return time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())


nowTime = getNowTime()


def getINFOLOG():
    return LogHandler(filename=f'log/filter-{nowTime}.log')


INFO_LOG = LogHandler(filename=f'log/filter-{nowTime}.log')


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        duration_dict = {}
        elapse = (time.perf_counter() - t) * 1000  # 单位为 ms
        if func.__name__ == 'adaptive_clipping':
            duration_dict['clustering'] = elapse
        elif func.__name__ == 'model_filtering_layer':
            duration_dict['clipping'] = elapse
        
        # print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun



def get_true_positive_rate(ccaf: list, client_poisoned_all: list, total_client: int = 10) -> float:
    '''
    description: compute TPR(True positive rate),
    param {int} ccaf: client_chooise_after_filter
    return {*}
    '''
    choose_poisoned_cnt = 0

    for client in ccaf:
        for kk in client_poisoned_all:
        #if client in client_poisoned_all:
            # print("type", int(client), int(kk))
            if int(client) == int(kk):
                choose_poisoned_cnt += 1
        # if client in client_poisoned_all:
        #     choose_poisoned_cnt += 1
    print("choose_poisoned_cnt", choose_poisoned_cnt)
    return (len(client_poisoned_all) - choose_poisoned_cnt) * 1.0 / (total_client - len(ccaf) + 1e-5), choose_poisoned_cnt, len(ccaf) - choose_poisoned_cnt


def get_true_negative_rate(ccaf: list, client_poisoned_all: list, total_client: int = 10) -> float:
    '''
    description: compute TNR(True negative rate)
    param {int} tpc: client_poisoned_all
    param {int} ccaf: client_chooise_after_filter
    return {*}
    '''

    rate = 0.0
    choose_poisoned_cnt = 0
    for client in ccaf:
        for kk in client_poisoned_all:
        #if client in client_poisoned_all:
            if int(client) == int(kk):
                choose_poisoned_cnt += 1
        # if client in client_poisoned_all:
        #     choose_poisoned_cnt += 1
    print("choose_poisoned_cnt", choose_poisoned_cnt)
    rate = (len(ccaf) - choose_poisoned_cnt) * 1.0 / (total_client - len(client_poisoned_all) + 1e-5)
    return rate, choose_poisoned_cnt, len(ccaf) - choose_poisoned_cnt


if __name__ == '__main__':
    logger = LogHandler.getINFOLogger()
