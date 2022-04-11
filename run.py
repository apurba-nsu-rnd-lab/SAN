import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
import utils, model
import math, random, argparse, statistics, datetime
from test import run_test

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", default=60, type=int, help="number of epochs per task")
parser.add_argument("-lr", default=0.001, type=float, help="learning rate for training")
parser.add_argument("-r", "--run", default=3, type=int, help="number of run")
args = parser.parse_args()

start_time = datetime.datetime.now().replace(microsecond=0)

# create task list and shuffle using random seed 1
task_class_list = utils.generate_task_class_list(n_cls=10, n_task=5, n_cls_per_task=2, verbose=True)

transform = transforms.Compose(
    [
     transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
    ])

batch_size = 32

root = os.path.join(os.getcwd(), 'datasets')
if not os.path.exists(root): os.makedirs(root)
    
trainset = torchvision.datasets.MNIST(root=root,train=True, download=True, transform=transform)

all_accuracy = []

for run in range(args.run):
    split = 0.15
    train_split, valid_split = torch.utils.data.random_split(
        trainset, [math.ceil(len(trainset)*(1-split)), math.floor(len(trainset)*split)]
    )

    # generate split data
    split_train_datasets = utils.generate_split_data(train_split, task_class_list)
    split_validaion_datasets = utils.generate_split_data(valid_split, task_class_list)

    ckpts_dir = os.path.join(os.getcwd(), 'ckpts')
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)

    # device, model/net, 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.EquivalentNetMNIST()

    # for each task
    for task in range(len(task_class_list)):
        if task==0:
            net = model.EquivalentNetMNIST()
        else:
            try:
                PATH = "./ckpts/task1.pth"
                net = model.EquivalentNetMNIST()
                net.load_state_dict(torch.load(PATH, map_location=device))

                for p in net.backbone.parameters():
                    p.requires_grad = False

                for p in net.classifier.parameters():
                    p.requires_grad = False

#                 for name, param in net.named_parameters(): 
#                     print(name, param.requires_grad)
            except:
                print('Model does not found')

        # data loaders
        trainloader = torch.utils.data.DataLoader(split_train_datasets[task], batch_size=batch_size,shuffle=True, num_workers=2 )
        validationloader = torch.utils.data.DataLoader(split_validaion_datasets[task], batch_size=batch_size, shuffle=True, num_workers=2)

        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

        # train model
        n_epoch = args.epoch
        utils.train(run, task, task_class_list, n_epoch, trainloader, validationloader, device, net, criterion, optimizer)
        
    run_accuracy = run_test()
    all_accuracy.append(run_accuracy)
    
end_time = datetime.datetime.now().replace(microsecond=0)
    
print()
print("Total time taken: ", end_time-start_time)


if len(all_accuracy)>1:
    print('----------------------------------------')
    print('\nAll Accuracies on Test set: ', all_accuracy)
    print('Average Test Accuracy: ', statistics.mean(all_accuracy))
    print('Standard Deviation: ', statistics.stdev(all_accuracy))

