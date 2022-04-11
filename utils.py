import random
import torch


def generate_task_class_list(n_cls, n_task, n_cls_per_task, verbose):
    # create task list and shuffle using random seed 1
    task = [x for x in range(n_cls)]
    random.Random(1).shuffle(task)
    
    task_class_list = []
    c = 0
    
    for i in range(n_task):
        task_classes = []
        for j in range(n_cls_per_task):
            task_classes.append(task[c])
            c+=1
        task_class_list.append(task_classes)
    
    if verbose:
        print("task_class_list:\n", task_class_list)
    
    return task_class_list


def generate_split_data(dataset, task_class_list):
    split_datasets = [[] for i in range(len(task_class_list))]

    for data in dataset:
        for i in range(len(task_class_list)):
            if data[1] in task_class_list[i]:
                split_datasets[i].append(data)

    return split_datasets

