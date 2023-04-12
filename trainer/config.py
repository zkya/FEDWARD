def setting(args):
  # args['IID'] = True
  if args['dataset'] == 'FASHION':
      print("FASHION DATASET")
      args['local_poison_epoch'] = 5
      args['epoch'] = 2
      args['attack_epoch_in'] = 5
      args['num_comm'] = 110
      args['gpu'] = '0'
  elif args['dataset'] == 'MNIST':
      print("MNIST DATASET")
      args['local_poison_epoch'] = 6
      args['epoch'] = 1
      args['attack_epoch_in'] = 5
      args['num_comm'] = 110
      args['gpu'] = '0'
  elif args['dataset'] == 'CIFAR':
      print("CIFAR DATASET")
      args['local_poison_epoch'] = 6
      args['epoch'] = 3
      args['attack_epoch_in'] = 10
      args['num_comm'] = 220
      args['gpu'] = '0'
  elif args['datasets'] == 'TINY':
      print("TINY DATASET")
      args['local_poison_epoch'] = 6
      args['epoch'] = 3
      args['attack_epoch_in'] = 10
      args['num_comm'] = 220
      args['gpu'] = '0'

  # if  args['IID_rate'] == 0:
  #     args['IID'] = False