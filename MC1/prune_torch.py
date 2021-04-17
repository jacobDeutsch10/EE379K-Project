import torch
import torch.nn as nn
from mobilenet_rm_filt_pt import *
from ptflops import get_model_complexity_info
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import json
from torchsummary import summary
import time



# Argument parser
parser = argparse.ArgumentParser()
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for finetuning
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
# Define the % of channels to prune
parser.add_argument('--fraction', type=float, default=0.2, help='fraction of channels to prune')
def main():
    
    args = parser.parse_args()


    # Always make assignments to local variables from your args at the beginning of your code for better
    # control and adaptability
    num_epochs = args.epochs
    batch_size = args.batch_size
    fraction = args.fraction
    experiment_name = f'e{num_epochs}_f{int(fraction*100)}'
    # load base, unpruned model
    model = MobileNetv1()
    model.load_state_dict(torch.load("mbnv1_pt.pt"))


    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print(f"Macs of unpruned: {macs}, #params of unpruned: {params}")


    model = channel_fraction_pruning(model,fraction)

    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
    print(f"Macs of \'pruned\': {macs}, #params of \'pruned\': {params}")

    model = remove_channel(model)
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print(f"Macs of pruned: {macs}, #params of pruned: {params}")
    stats = {'Macs': macs, 'params': params, 'train_acc': [], 'test_acc':[], 'time': 0}
    random_seed = 1
    torch.manual_seed(random_seed)

    # CIFAR10 Dataset (Images and Labels)
    train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]), download=True)

    test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]))

    # Dataset Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # TODO: Put the model on the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    # Define your loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.Adam(model.parameters())
    global_step = 0
    

    for epoch in range(num_epochs):
        start = time.time()
        # Training phase loop
        train_correct = 0
        train_total = 0
        train_loss = 0
        # Sets the model in training mode.
        #model = model.train()
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # TODO: Put the images and labels on the GPU
            images, labels = images.to(device), labels.to(device)
            # Sets the gradients to zero
            optimizer.zero_grad()
            # The actual inference
            outputs = model(images)
            # Compute the loss between the predictions (outputs) and the ground-truth labels
            loss = criterion(outputs, labels)
            # Do backpropagation to update the parameters of your model
            loss.backward()
            # Performs a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            # Print every 100 steps the following information
            global_step += 1
            if (batch_idx + 1) % 100 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                                len(train_dataset) // batch_size,
                                                                                train_loss / (batch_idx + 1),
                                                                                100. * train_correct / train_total))
                stats['train_acc'].append((epoch, global_step, 100. * train_correct / train_total))
                stats['time'] +=  time.time() - start
        # Testing phase loop
        test_correct = 0
        test_total = 0
        test_loss = 0
        # Sets the model in evaluation mode
        model = model.eval()
        # Disabling gradient calculation is useful for inference.
        # It will reduce memory consumption for computations.
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # TODO: Put the images and labels on the GPU
                images, labels = images.to(device), labels.to(device)
                # Perform the actual inference
                outputs = model(images)
                # Compute the loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                # The outputs are one-hot labels, we need to find the actual predicted
                # labels which have the highest output confidence
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))
        stats['test_acc'].append((epoch, global_step, 100. * test_correct / test_total))
    if num_epochs == 0:
        # Testing phase loop
        test_correct = 0
        test_total = 0
        test_loss = 0
        # Sets the model in evaluation mode
        model = model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # TODO: Put the images and labels on the GPU
                images, labels = images.to(device), labels.to(device)
                # Perform the actual inference
                outputs = model(images)
                # Compute the loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                # The outputs are one-hot labels, we need to find the actual predicted
                # labels which have the highest output confidence
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
            print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))
            stats['test_acc'].append((0, global_step, 100. * test_correct / test_total))
    with open(f"stats_train_{experiment_name}.json", "w") as f:
        f.write(json.dumps(stats))
        f.write('\n')
    model.to('cpu')
    torch.onnx.export(model,torch.rand(1,3,32,32), f"mbnv1_{experiment_name}.onnx",export_params=True,opset_version=10)

if __name__ == "__main__":
    main()