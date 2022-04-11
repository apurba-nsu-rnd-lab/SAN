import torch

def validate(task, task_class_list, testloader, device, net, epoch, criterion, test=False):
    net.eval()
    valid_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, o_labels = data    
            labels = []
            for label in o_labels:
                labels.append(task_class_list[task].index(label)) 
            labels = torch.as_tensor(labels)

            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            valid_running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = valid_running_loss/len(testloader)
    accuracy = 100 * correct / total

    if test:
        print('Test_loss: {test_l: .4f}, Test_Accuracy: {test_a: .2f}'.format(
        test_l=epoch_loss, test_a=accuracy))
    else:
        print('Epoch: {}, Valid_loss: {vl: .4f}, Valid_Accuracy: {va: .2f}'.format(
        epoch+1, vl=epoch_loss, va=accuracy))
        
    return epoch_loss, accuracy


def train(run, task, task_class_list, n_epoch, trainloader, validationloader, device, net, criterion, optimizer):
    print()
    print()
    print("**************** RUN {}, TASK {} ***************".format(run+1, task+1))

    best_val_loss = float('inf')
    for epoch in range(n_epoch):  
        net.train()
        net.to(device)
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader):
            inputs, o_labels = data 
            labels = [] 
            for label in o_labels:
                labels.append(task_class_list[task].index(label))  #get current index of the class
            labels = torch.as_tensor(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss/len(trainloader)
        accuracy = 100 * correct / total
        print('Epoch: {}, Train_loss: {tl: .4f}, Train_Accuracy: {ta: .2f}'.format(
                    epoch+1, tl=epoch_loss, ta=accuracy)
            )
        
        # validate and save the best model
        val_loss, val_acc = validate(task, task_class_list, validationloader, device, net, epoch, criterion, test=False)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            PATH = './ckpts/task{}.pth'.format(task+1)
            torch.save(net.state_dict(), PATH)
            print('Best Validation result in Epoch: ', epoch+1)
        print()


    
