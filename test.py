import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import utils, engine, models


def run_test(exp_id):
    all_acc = []
    # create 20 split and assign classes to each task(split)
    task_class_list = utils.get_task_class_list(exp_id)

    batch_size = 32

    transform = utils.get_transform(exp_id=exp_id)

    root = os.path.join(os.getcwd(), 'datasets')
    testset = utils.get_testset(root=root, exp_id=exp_id, transform=transform)

    # generate split data
    split_test_datasets = utils.generate_split_data(testset, task_class_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    

    total_accuracy = 0

    for task in range(len(task_class_list)):
        print("Task:", task+1)
        PATH = "ckpts/{}/task{}.pth".format(utils.exp_dict[exp_id], task+1)
        net = utils.get_model(exp_id).to(device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--experiment", default=1, type=int, help="id of the experiment.")
    args = parser.parse_args()
    
    run_test(exp_id=args.experiment)