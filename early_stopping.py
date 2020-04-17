import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=30,learning_rate=0.1,verbose=False):
        """  Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 30
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.learning_rate = learning_rate

        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        LR_9f = '%.9f' % self.learning_rate
        print('LR_9f',float(LR_9f))
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter ==15:
                self.learning_rate /= 10
            elif self.counter ==  30  or float(LR_9f) == 1e-9:
                # torch.save(model, './classification_process/model_best_classification_process.pkl')   # few shot heamorrhage with weak label
                torch.save(model, './segmentation_process/model_last_segmentation_process.pkl.pkl')   # few shot heamorrhage with weak label
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience or float(LR_9f) == 1e-9 :
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.learning_rate

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model, './classification_process/model_best_classification_process.pkl')   # few shot heamorrhage with weak label
        torch.save(model, './segmentation_process/model_best_segmentation_process.pkl')   # few shot heamorrhage with weak label
        self.val_loss_min = val_loss