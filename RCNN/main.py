import os
import torch

from torch.utils.data import DataLoader

import utils

from Classifier import get_classifier
from Dataset import CustomVOCDataset

# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

def sanity_check(loader, model):
    images, targets = next(iter(loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)

    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions
    print(predictions[0])

from engine import train_one_epoch

def run(model, train_loader, val_loader, device):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it for 5 epochs
    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        model.evaluate(val_loader, epoch, 'validation')

    print("That's it!")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    concepts = ['Asteroidea',
                'Bivalvia',
                'Ceriantharia',
                'Crinoidea',
                'Gastropoda',
                'Hexacorallia',
                'Holothuroidea',
                'Octocorallia',
                'Ophiuroidea',
                'Porifera',
                'Pycnogonida']
    model = get_classifier(concepts, device)
    model.to(device)

    train_dataset = CustomVOCDataset(concepts, 'train')
    triain_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    val_dataset = CustomVOCDataset(concepts, 'test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    #sanity_check(triain_loader, model)
    run(model, triain_loader, val_loader, device)


## TODO-KM we still want to add transforms

if __name__ == '__main__':
    main()