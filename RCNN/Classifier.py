import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

def get_classifier(concepts, device):
    return BenthicResNetClassifier(concepts, device)


class BenthicResNetClassifier(torch.nn.Module):
    # https://github.com/dhruvbird/ml-notebooks/blob/main/Flowers-102-transfer-learning/flowers102-classification-using-pre-trained-models.ipynb
    def __init__(self, concepts, device='cpu'):
        super().__init__()
        self.new_layers = []
        self.best_accuracy = 0.0
        self.true_batch_size = 500

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features        
        new_classifier = FastRCNNPredictor(
            in_features, len(concepts) + 1
        )

        self.new_layers = [new_classifier]
        self.model.roi_heads.box_predictor = new_classifier

        self.dummy_param = torch.nn.Parameter(torch.empty(0)).to(device=device)
            

    def forward(self, image, targets):
        return self.model(image, targets)
    
    def fine_tune(self, extract=True):
        # The requires_grad parameter controls whether this parameter is
        # trainable during model training.
        m = self.model
        for p in m.parameters():
            p.requires_grad = False

        if extract:
            for l in self.new_layers:
                for p in l.parameters():
                    p.requires_grad = True
                    
        else:
            for p in m.parameters():
                p.requires_grad = True

    def train_one_epoch(self, optimizer, data_loader, scheduler=None, epoch=1):
        """Train this model for a single epoch. Return the loss computed
        during this epoch.
        """
        device = self.dummy_param.device
        running_loss = 0.0
        num_batches = 0
    
        for images, targets in data_loader:
            images = list(image.to(device=device, dtype=torch.float) for image in images)
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]


            with torch.cuda.amp.autocast():            
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = {}
            for loss_key, loss_item in loss_dict.items():
                loss_dict_reduced[loss_key] = loss_item.sum().item()
            
            loss_value = sum(loss for loss in loss_dict_reduced.values())
            optimizer.zero_grad()
            
            losses.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                        
            running_loss, num_batches = running_loss + loss_value, num_batches + 1

        print(f"[{epoch}] Train Loss: {running_loss / num_batches:0.5f}")
        return running_loss / num_batches

    def evaluate(self, data_loader, epoch, run_type):
        """Evaluate the model on the specified dataset (provided using the DataLoader
        instance). Return the loss and accuracy.
        """
        device = self.dummy_param.device

        iou_values = []
        for images, targets in data_loader:
            images = list(image.to(device=device, dtype=torch.float) for image in images)
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
        
            with torch.no_grad():
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                predictions = self.model(images, targets)
        
            for prediction, target in zip(predictions, targets):
                predicted_boxes = prediction['boxes']
                target_boxes = target['boxes']
                iou = box_iou(predicted_boxes, target_boxes)
                iou_values.append(iou)
            
        mean_iou = torch.cat(iou_values).mean().item()
        print(f"Mean {run_type} IoU epoch {epoch}: {mean_iou}")
        return mean_iou

    def train_multiple_epochs_and_save_best_checkpoint(
        self,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs,
        filename,
        training_run,
    ):
        """Train this model for multiple epochs. The caller is expected to have frozen
        the layers that should not be trained. We run training for "epochs" epochs.
        The model with the best val accuracy is saved after every epoch.
        
        After every epoch, we also save the train/val loss and accuracy.
        """
        for epoch in range(1, epochs + 1):
            self.train()
            train_loss = self.train_one_epoch(optimizer, train_loader, scheduler, epoch)

            # Evaluate accuracy on the train dataset.
            self.eval()            
            with torch.inference_mode():
                train_acc = self.evaluate(train_loader, epoch, "Train")
                training_run.train_loss.append(train_loss)
                training_run.train_iou.append(train_acc)

            # Evaluate accuracy on the val dataset.
            self.eval()
            with torch.inference_mode():
                val_acc = self.evaluate(val_loader, epoch, "Val")
                training_run.val_iou.append(val_acc)
                if val_acc > self.best_accuracy:
                    print(f"Current valdation accuracy {val_acc*100.0:.2f} is better than previous best of {self.best_accuracy*100.0:.2f}. Saving checkpoint.")
                    torch.save(self.state_dict(), filename)
                    self.best_accuracy = val_acc
            
            scheduler.step()

    def get_optimizer_params(self):
        """This method is used only during model fine-tuning when we need to
        set a linear or expotentially decaying learning rate (LR) for the
        layers in the model. We exponentially decay the learning rate as we
        move away from the last output layer.
        """
        options = []

        layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
        lr = 0.0001
        for layer_name in reversed(layers):
            options.append({
                "params": getattr(self.model, layer_name).parameters(),
                'lr': lr,
            })
            lr = lr / 3.0

        return options