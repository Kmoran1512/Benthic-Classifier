import matplotlib.pyplot as plt

class TrainingRun:
    def __init__(self):
        self.train_loss = []
        self.train_iou = []
        self.val_loss = []
        self.val_iou = []
        self.test_iou = []
    
    def plot(self, plt, ax):
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        epochs = list(range(1, len(self.train_loss) + 1))

        ax.plot(epochs, self.train_loss, label="Train Loss", c="blue")
        ax.plot(epochs, self.val_loss, label="Val Loss", c="orange")
        plt.legend()

        ax2 = ax.twinx()
        ax2.set_ylabel("Intersection over Union (IoU)")

        ax2.plot(epochs, self.val_iou, label="Val IoU", c="green")
        if len(self.train_iou) == len(epochs):
            ax2.plot(epochs, self.train_iou, label="Train IoU", c="cyan")

        plt.legend()
