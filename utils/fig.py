import torch
import matplotlib.pyplot as plt


def draw_mnist(path):
    dataset = "CIFAR"
    filename = f"../tmp/" + path
    print(filename)
    record = torch.load(filename)
  
    main_acc_global = record[1]
    backdoor_acc = record[2]
    defend_rate = record[3]
    print("Record", backdoor_acc)

    plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1)
    plt.plot(main_acc_global)
    if dataset == "FASHION":
        plt.ylim(0.7, 1)
    plt.xlabel("round")
    plt.title("main_acc_global")

    plt.subplot(1, 4, 2)
    for i in range(4):
        plt.plot(torch.tensor(backdoor_acc[i + 1], device='cpu'))
    plt.ylim(0, 0.5)
    plt.title("four_trigger_acc")

    plt.subplot(1, 4, 3)
    backdoor_acc_global = torch.tensor(backdoor_acc[0], device='cpu')
    plt.plot(backdoor_acc_global)
    plt.ylim(0, 0.5)
    plt.title("backdoor_acc_global")

    plt.subplot(1, 4, 4)
    plt.plot(defend_rate)
    plt.title("backdoor_select_rate")
    # plt.ylim(0, 100)
    plt.savefig("../figs/" + path + ".png")
    # plt.show()


def draw_fashion(path):
    dataset = "FASHION"
    filename = f"../tmp/" + path
    print(filename)
    record = torch.load(filename)
  
    main_acc_global = record[1]
    backdoor_acc = record[2]
    defend_rate = record[3]
    print("Record", backdoor_acc)

    plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1)
    plt.plot(main_acc_global)
    if dataset == "FASHION":
        plt.ylim(0.7, 1)
    plt.xlabel("round")
    plt.title("main_acc_global")

    plt.subplot(1, 4, 2)
    for i in range(4):
        plt.plot(torch.tensor(backdoor_acc[i + 1], device='cpu'))
    plt.ylim(0, 0.5)
    plt.title("four_trigger_acc")

    plt.subplot(1, 4, 3)
    backdoor_acc_global = torch.tensor(backdoor_acc[0], device='cpu')
    plt.plot(backdoor_acc_global)
    plt.ylim(0, 0.5)
    plt.title("backdoor_acc_global")

    plt.subplot(1, 4, 4)
    plt.plot(defend_rate)
    plt.title("backdoor_select_rate")
    plt.ylim(0, 1)
    plt.savefig("../figs/" + path+ ".png")


def draw(dataset, path):
    poison_rate = [0.15625, 0.3125]
    # for j in range(2):
    for i in range(3):
        filename = f"../tmp/" + path
        print(filename)
        record = torch.load(filename)
        main_acc_global = record[0]
        backdoor_acc = record[1]
        defend_rate = record[2]

        plt.figure(figsize=(24, 5))
        plt.subplot(1, 4, 1)
        plt.plot(main_acc_global)
        if dataset == "FASHION":
            plt.ylim(0.7, 1)
        plt.xlabel("round")
        plt.title("main_acc_global")

        plt.subplot(1, 4, 2)
        for i in range(4):
            plt.plot(torch.tensor(backdoor_acc[i + 1], device='cpu'))
        plt.ylim(0, 0.5)
        plt.title("four_trigger_acc")

        plt.subplot(1, 4, 3)
        backdoor_acc_global = torch.tensor(backdoor_acc[0], device='cpu')
        plt.plot(backdoor_acc_global)
        plt.ylim(0, 0.5)
        plt.title("backdoor_acc_global")

        plt.subplot(1, 4, 4)
        plt.plot(defend_rate)
        plt.title("backdoor_select_rate")
        # plt.ylim(0, 100)
        plt.show()

def draw_two(path1, path2, dataset = 'MNIST'):
    poison_rate = [10 / 64]
    size = len(poison_rate)
    plt.figure(figsize=(24, 5 * size))
    fontsize = 20
    for i in range(size):
        filename = f"../tmp/" + path1
        filename2 = f"../tmp/" + path2
        record_cos = torch.load(filename)
        record_hamming = torch.load(filename2)

        plt.subplot(size, 4, i * 4 + 1)
        plt.plot(record_cos[1], label='cos')
        plt.plot(record_hamming[0], label='eud')
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("main_acc_global", fontsize=fontsize)
        plt.legend()

        backdoor_acc = record_hamming[1]
        plt.subplot(size, 4, i * 4 + 2)
        for j in range(4):
            plt.plot(backdoor_acc[j + 1])
        plt.ylim(0, 0.5)
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("four_trigger_acc", fontsize=fontsize)
        plt.subplot(size, 4, i * 4 + 3)
        plt.plot(record_cos[2][0], label='cos')
        plt.plot(record_hamming[1][0], label='eud')
        plt.ylim(0, 0.5)
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("backdoor_acc_global", fontsize=fontsize)
        plt.legend()

        plt.subplot(size, 4, i * 4 + 4)
        plt.plot(record_cos[3], linewidth=0, marker='o', markersize=6, label='cos')
        plt.plot(record_hamming[2], linewidth=0, marker='<', markersize=6, label='eud')
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("backdoor_select_rate", fontsize=fontsize)
        plt.legend()

    plt.savefig(f"../figs/compare.png")
    plt.show()


def draw2(dataset, path1, path2):
    # poison_rate = [5 / 64, 10 / 64, 20 / 64]
    poison_rate = [10 / 64]
    client = 50
    size = len(poison_rate)
    plt.figure(figsize=(24, 5 * size))
    fontsize = 20
    for i in range(size):
        filename = f"../tmp/client-{client}/" + path1
        filename2 = f"../tmp/client-{client}/" + path2
        record_cos = torch.load(filename)
        record_hamming = torch.load(filename2)

        plt.subplot(size, 4, i * 4 + 1)
        plt.plot(record_cos[0], label='cos')
        plt.plot(record_hamming[0], label='eud')
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("main_acc_global", fontsize=fontsize)
        plt.legend()

        backdoor_acc = record_hamming[1]
        plt.subplot(size, 4, i * 4 + 2)
        for j in range(4):
            plt.plot(backdoor_acc[j + 1])
        plt.ylim(0, 0.5)
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("four_trigger_acc", fontsize=fontsize)
        plt.subplot(size, 4, i * 4 + 3)
        plt.plot(record_cos[1][0], label='cos')
        plt.plot(record_hamming[1][0], label='eud')
        plt.ylim(0, 0.5)
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("backdoor_acc_global", fontsize=fontsize)
        plt.legend()

        plt.subplot(size, 4, i * 4 + 4)
        plt.plot(record_cos[2], linewidth=0, marker='o', markersize=6, label='cos')
        plt.plot(record_hamming[2], linewidth=0, marker='<', markersize=6, label='eud')
        if i == size - 1:
            plt.xlabel("round", fontsize=fontsize)
        if i == 0:
            plt.title("backdoor_select_rate", fontsize=fontsize)
        plt.legend()

    plt.savefig(f"../figs/png/{dataset}-2.png")
    plt.show()


# draw("FASHION")
# draw("MNIST")
# draw2("CIFAR")


def draw3(filename, save_path, title, start_attack=100, scheme="eflame"):
    print(f'draw figure from {filename}')

    record = torch.load(filename)
    if scheme == "flame":
        main_acc_global = record[1]
        backdoor_acc = record[2]
        tpr = record[3]
        tnr = record[4]
    elif scheme == "eflame":
        main_acc_global = record[0]
        backdoor_acc = record[1]
        tpr = record[2]
        tnr = record[3]

    plt.figure(figsize=(16, 4), )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    plt.subplot(1, 4, 1)

    plt.plot(main_acc_global, label='main_acc_global')
    plt.axvline(start_attack, linestyle="--", color="red")
    for x in range(0, len(main_acc_global), len(main_acc_global) // 6):
        plt.scatter(x, main_acc_global[x], marker='o')
        plt.text(x, main_acc_global[x] - 0.03, '%.4f' % main_acc_global[x], fontdict={'fontsize': 8},
                 verticalalignment='center',
                 horizontalalignment='center')
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.title("main_acc_global")
    # plt.legend()

    # plt.subplot(1, 4, 2)
    # for i in range(4):
    #     plt.plot(torch.tensor(backdoor_acc[i + 1], device='cpu'), label='backdoor_acc_' + str(i + 1))
    # plt.ylim(0, 0.3)
    # plt.xlabel("epoch")
    # plt.title("four_trigger_acc")
    # plt.legend()

    plt.subplot(1, 4, 2)
    plt.axvline(start_attack, linestyle="--", color="red")
    backdoor_acc_global = torch.tensor(backdoor_acc[0], device='cpu')
    backdoor_acc_global[:start_attack] = 0
    plt.plot(backdoor_acc_global, label='backdoor_acc_global')
    for x in range(0, len(backdoor_acc_global), len(backdoor_acc_global) // 6):
        if x < start_attack: continue
        plt.scatter(x, backdoor_acc_global[x], marker='o')
        plt.text(x, backdoor_acc_global[x] + 0.03, '%.4f' % backdoor_acc_global[x], fontdict={'fontsize': 6},
                 verticalalignment='center',
                 horizontalalignment='center')
    plt.ylim(0, 1.03)
    plt.xlabel("epoch")
    plt.title("backdoor_acc_global", )
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.axvline(start_attack, linestyle="--", color="red")
    plt.plot(tpr, label='TPR', linewidth=0, marker='<', markersize=3, color='green')

    plt.xlabel("epoch")
    plt.title("TPR")
    plt.legend()
    plt.ylim(-0.03, 1.03)

    plt.subplot(1, 4, 4)
    plt.axvline(start_attack, linestyle="--", color="red")
    plt.title("TNR")
    plt.xlabel("epoch")
    plt.plot(tnr, label='TNR', linewidth=0, marker='<', markersize=3, color='green')
    plt.legend()
    plt.ylim(-0.03, 1.03)

    plt.suptitle(title)
    plt.savefig(save_path)
    plt.show()
    print(f"the result draw in {save_path}")


def draw_single(file_path, save_path, title):
    record = torch.load(file_path)
    train_acc = record[0]
    test_acc = record[1]
    plt.plot(train_acc, label='train_acc', color="red")
    plt.plot(test_acc, label='test_acc', color="green")
    for x in range(0, len(train_acc)):
        if x % 9 == 0 or x == len(train_acc) - 1:
            plt.scatter(x, train_acc[x], marker='o')
            plt.scatter(x, test_acc[x], marker='o')

            flag = 0.03 if train_acc[x] > test_acc[x] else -0.03

            plt.text(x, train_acc[x] + flag, '%.2f' % train_acc[x], fontdict={'fontsize': 8},
                     verticalalignment='center',
                     horizontalalignment='center', )
            plt.text(x, test_acc[x] - flag, '%.2f' % test_acc[x], fontdict={'fontsize': 8}, verticalalignment='center',
                     horizontalalignment='center', )
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig(save_path)
    print(f"the result draw in {save_path}")


if __name__ == '__main__':
    # file_name = 'tmp/eflame/MNIST-2022-08-14 21_26_37-posion_rate(0.078125).pth'
    # save_path = 'img/com/eflame/MNIST-2022-08-14 21_26_37-posion_rate(0.078125).png'
    # title = 'MNIST-2022-08-14 21_26_37-posion_rate(0.078125)'
    # draw3(file_name, save_path, title, start_attack=20, scheme="eflame")

    # file_path = "tmp/pretrain/CIFAR_acc.pth"
    # save_path = "img/pretrain/CIFAR_acc_resnet.png"
    # title = "CIFAR_acc_resnet"
    # draw_single(file_path, save_path, title)
    # draw_mnist()
    # draw_two()
    #aa = torch.load('tmp/daguard/CIFAR/0.25-ternGrad-CIFAR-2022-12-13 23_55_18-4-posion_rate(0.46875).pth')
    name = '/0.25-ternGrad-FASHION-2022-12-17 13_06_59-4-posion_rate(0.3125).pth'
    flame = 'tmp/flame_result/'
    dagurad = 'tmp/daguard/'
    dataset = 'FASHION'
    aa = torch.load(dagurad + dataset + name)
    lst = aa[1][0][10:]
    print("aa", sum(lst)/len(lst))
    # draw_fashion()