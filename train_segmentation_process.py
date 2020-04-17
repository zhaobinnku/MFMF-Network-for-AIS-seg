from early_stoping import EarlyStopping
from torch.utils.data import  DataLoader
from tqdm import tqdm
import visdom
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.backends import cudnn
from radam import RAdam
from MFMF_Network import *
import argparse


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.determinstic = True
#
setup_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
model = segmentaion_process()

model = model.cuda()
parser = argparse.ArgumentParser(description='Segmentation process')
print(parser.description)
traindata_dir = './my_train_data_seg/'

parser.add_argument('--train_img_dir', default=traindata_dir, help='path to directory containing the images')
parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Enable visualization(s) on visdom | False by default')
parser.add_argument('--no-visdom', dest='visdom', action='store_false',
                    help='Disable visualization(s) on visdom | False by default')
parser.set_defaults(visdom=True)
opts = parser.parse_args()

train_dataset = train_dataset_segmentation(img_dir=opts.train_img_dir, transforms=True)
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, drop_last=False, shuffle=True)

if opts.visdom:
    vis = visdom.Visdom()
    train_loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                                 Y=torch.zeros((1)).cpu(),
                                 opts=dict(xlabel='epoch',
                                           ylabel='Loss',
                                           title='Training Loss',
                                           legend=['Loss']))

optimizer = RAdam(filter(lambda p: p.requires_grad,model.parameters()), lr=opts.learning_rate)
criterion = nn.BCELoss()
cudnn.benchmark = True
train_add_loss = []
train_epoch = []
one_epoch_iteration = len(train_dataloader)
early_stoping = EarlyStopping(patience=30, learning_rate=opts.learning_rate, verbose=True)

for epoch in tqdm(range(opts.epochs)):
    train_epoch.append(epoch)
    np.savetxt('data_experiment/train_epoch', train_epoch)
    model.train()
    train_loss = 0.0

    for index, sample in enumerate(train_dataloader):
        img = Variable(sample['image'].float()).cuda()
        label = Variable(sample['label'].float()).cuda()

        label_numpy = label.data.cpu().numpy()
        prediction_prob = model(img)
        loss = criterion(prediction_prob, label)

        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        prediction = prediction_prob.data.cpu().numpy()
        label = label.data.cpu().numpy()

        print('training_loss: {:.4f}'.format(loss.item()))
    print('train_loss: {:.6f} '.format((train_loss / one_epoch_iteration)))
    train_add_loss.append(train_loss / one_epoch_iteration)
    np.savetxt('data_experiment/train_add_loss', train_add_loss)

    new_learning_rate = early_stoping((train_loss / one_epoch_iteration), model)
    optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.learning_rate)
    print('new_lr', new_learning_rate)
    if early_stoping.early_stop:
        print("Early stoping")
        break
    if opts.visdom:
        vis.line(X=torch.ones((1,)).cpu() * epoch,
                 Y=torch.Tensor([train_loss / one_epoch_iteration]).unsqueeze(0).cpu(),
                 win=train_loss_window,
                 update='append')
plt.figure(1)
plt.plot(train_epoch, train_add_loss, '-ob')
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.title('train_loss_curve')
plt.show()

