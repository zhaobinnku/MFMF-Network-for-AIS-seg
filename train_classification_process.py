from classification_process_dataset import *
from early_stoping import EarlyStopping
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import visdom
import torchvision.models as md
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.backends import cudnn
from radam import RAdam
from vgg16 import VGG16
import argparse
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.determinstic = True
#
setup_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"]="1"


model = VGG16()        # train VGG16_net
vgg16 = md.vgg16_bn(pretrained=True)
model.init_vgg16_params(vgg16)
model.features1[0] = torch.nn.Conv2d(2, 64, kernel_size=3, padding=1)

model = model.cuda()
parser = argparse.ArgumentParser(description='Classification process')

traindata_dir = './my_train_data_cls/'
validation_data_dir = './my_validation_data_cls/'

parser.add_argument('--train_img_dir', default=traindata_dir, help='path to directory containing the images')
parser.add_argument('--validation_img_dir', default=validation_data_dir, help='path to directory containing the images')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                        help='Enable visualization(s) on visdom | False by default')
parser.add_argument('--no-visdom', dest='visdom', action='store_false',
                        help='Disable visualization(s) on visdom | False by default')
parser.set_defaults(visdom=True)
opts = parser.parse_args()
train_dataset=train_strokelesion_Dataset(img_dir=opts.train_img_dir, transform=True)
train_dataloader=DataLoader(train_dataset,batch_size=opts.batch_size,drop_last=False,shuffle=True)

validation_dataset=validation_strokelesion_Dataset(img_dir=opts.validation_img_dir)
validation_dataloader=DataLoader(validation_dataset,batch_size=opts.batch_size,drop_last=False,shuffle=True)
if opts.visdom:
    vis = visdom.Visdom()
    train_loss_window = vis.line(X=torch.zeros((1,)).cpu(),
    Y=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch',
              ylabel='Loss',
              title='Training Loss',
              legend=['Loss']))
    train_acc_window_AIS = vis.line(X=torch.zeros((1,)).cpu(),
    Y=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch',
              ylabel='Accuracy',
              title='Training Accuracy_AIS',
              legend=['Accuracy']))
    validation_loss_window = vis.line(X=torch.zeros((1,)).cpu(),
    Y=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch',
              ylabel='Loss',
              title='Validation Loss',
              legend=['Loss']))

try:
    model = torch.load('./model/model.pkl')
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")
    pass

optimizer = RAdam(list(model.parameters()),lr=opts.learning_rate)
criterion = nn.BCELoss()
cudnn.benchmark = True
train_add_loss = []
train_correct = []
train_epoch_loss = []
train_iteration = []
train_epoch = []
train_learning_rate = []
validation_correct = []
validation_add_loss = []
one_epoch_iteration = len(train_dataloader)
early_stoping = EarlyStopping(patience=30,learning_rate=opts.learning_rate,  verbose=True)
for epoch in tqdm(range(opts.epochs)):
    train_epoch.append(epoch)
    np.savetxt('data_experiment/train_epoch', train_epoch)
    model.train()
    # epoch_loss = 0
    train_loss = 0

    train_acc = 0
    minibatch = 0
    for index, sample in enumerate(train_dataloader):

        img = Variable(sample['image'].float()).cuda()
        label = Variable(sample['label'].float()).cuda()
        label = label.unsqueeze(dim=1)
        prediction_prob = model(img)

        # AIS loss
        loss = criterion(prediction_prob, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        prediction = prediction_prob > 0.5
        train_epoch_loss.append(loss.item())
        np.savetxt('data_experiment/train_epoch_loss', train_epoch_loss)
        label = label.cpu().numpy()
        prediction = prediction.cpu().numpy()
        correct = sum((label == prediction))
        train_acc += correct.item()


    print('\n')
    print ('train_loss: {:.6f} '.format((train_loss / one_epoch_iteration)))
    print ('train_acc_AIS: {:.6f} '.format((train_acc / len(train_dataset))))

    train_correct.append(train_acc / len(train_dataset))
    np.savetxt('data_experiment/train_correct', train_correct)
    train_add_loss.append(train_loss / one_epoch_iteration)
    np.savetxt('data_experiment/train_add_loss', train_add_loss)

    model.eval()
    eval_loss = 0
    eval_acc = 0
    for index, sample in enumerate(validation_dataloader):
        img = Variable(sample['image'].float()).cuda()
        label = Variable(sample['label'].float()).cuda()
        label = label.unsqueeze(dim=1)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()

        prediction = out > 0.5
        label = label.cpu().numpy()
        prediction = prediction.cpu().numpy()
        correct = sum((label == prediction))
        eval_acc += correct.item()
    print("validation_loss: {:.6f} ".format(eval_loss / (len(validation_dataloader))))
    print("validation_acc_AIS: {:.6f}".format((eval_acc / len(validation_dataset))))
    validation_correct.append(eval_acc / len(validation_dataset))
    np.savetxt('data_experiment/validation_correct', validation_correct)
    validation_add_loss.append(eval_loss / len(validation_dataloader))
    np.savetxt('data_experiment/validation_add_loss', validation_add_loss)
    new_learning_rate = early_stoping(eval_loss / (len(validation_dataloader)), model)
    optimizer = RAdam(list(model.parameters()), lr=new_learning_rate)
    print('new_lr', new_learning_rate)
    if early_stoping.early_stop:
        print("Early stoping")
        break
    if opts.visdom:
        vis.line(X=torch.ones((1, )).cpu() * epoch,
                 Y=torch.Tensor([train_loss / one_epoch_iteration]).unsqueeze(0).cpu(),
                 win=train_loss_window,
                 update='append')
        vis.line(X=torch.ones((1, )).cpu() * epoch,
                 Y=torch.Tensor([train_acc / len(train_dataset)]).cpu(),
                 win=train_acc_window_AIS,
                 update='append')
        vis.line(X=torch.ones((1,)).cpu() * epoch,
                 Y=torch.Tensor([eval_loss / len(validation_dataloader)]).unsqueeze(0).cpu(),
                 win=validation_loss_window,
                 update='append')



plt.figure(1)
plt.plot(train_epoch, train_add_loss, '-ob')
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.title('train_loss_curve')
plt.figure(2)
plt.plot(train_epoch, np.array(train_correct), '-+r')
plt.xlabel('epoch')
plt.ylabel('train_correct')
plt.title('train_correct_curve')
plt.figure(3)
plt.plot(train_epoch, validation_add_loss, '--*g')
plt.xlabel('epoch')
plt.ylabel('validation_loss')
plt.title('validation_loss_curve')
plt.figure(4)
plt.plot(train_epoch, np.array(validation_correct), '--Dk')
plt.xlabel('epoch')
plt.ylabel('validation_correct')
plt.title('validation_correct_curve')
plt.show()
        
        