import torch
import torch.nn as nn
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet =models.squeezenet1_0()


#pretrained=True就可以使用预训练的模型
#resnet18 = models.resnet18(pretrained=True)


#是否适用已经训练好的模型特征
feature_extract = True

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CPU')
else:
    print('GPU')    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_parameter_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

print(resnet18)    


def initialize_model(model_name,num_classes,feature_extract,use_pretained=True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet':

        model_ft = models.resnet152(pretrained=use_pretained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs,102),
            nn.LogSoftmax(dim=1)
        )
        input_size = 224

    elif model_name == "alexnet":

        model_ft = models.alexnet(pretrained=use_pretained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":

        model_ft = models.vgg16(pretrained=use_pretained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_ftrs)
        input_size = 224

    return model_ft,input_size


model_ft,input_size = initialize_model(model_name,102,feature_extract,use_pretained=True)

filename = "checkpoint.pth"
params_to_update =model_ft.parameters()
print('Params to learn:')
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)            


optimizer_ft = optim.Adam(params_to_update,lr=1e-2)
scheduler = optim.lr.scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
criterion = nn.NLLoss()

def train_model(model,dataloaders,criterion,optimizer,num_epochs=25,is_inception=False,filename=filename):
    since = time.time()
    best_acc = 0

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs);
    print("Epoch {}/{}".format(epoch,num_epochs))

#标准化
from sklearn import preprocessing
data = preprocessing.StandardScalar().fittransform(features)

#通道转换位置
img = img.transpose((2,0,1))
