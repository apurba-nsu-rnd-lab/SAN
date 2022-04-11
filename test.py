import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import utils, engine, model


def run_test():
    all_acc = []
    # create 20 split and assign classes to each task(split)
    task_class_list = utils.generate_task_class_list(n_cls=10, n_task=5, n_cls_per_task=2, verbose=False)

    batch_size = 32

    transform = transforms.Compose(
        [
         transforms.Grayscale(1),
         transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))
        ])

    root = os.path.join(os.getcwd(), 'datasets')
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # generate split data
    split_test_datasets = utils.generate_split_data(testset, task_class_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    

    total_accuracy = 0

    for task in range(len(task_class_list)):
        print("Task:", task+1)
        PATH = "ckpts/task{}.pth".format(task+1)
        net = model.EquivalentNetMNIST().to(device)
        net.load_state_dict(torch.load(PATH, map_location=device))
        net.eval()
        testloader = torch.utils.data.DataLoader(split_test_datasets[task], batch_size=batch_size, shuffle=False, num_workers=2)
        task_loss, task_acc = engine.validate(task, task_class_list, testloader, device, net, 0, criterion, test=True)
        total_accuracy += task_acc
        all_acc.append(task_acc)

    avg_accuracy = total_accuracy/len(task_class_list)

    print()
    print()
    print("Average Accuracy: ", avg_accuracy)
    print('Individual Task Accuracy:', all_acc)
    return avg_accuracy

if __name__ =='__main__':
    run_test()