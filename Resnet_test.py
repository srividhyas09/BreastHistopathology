from ResNet_Preprocessing import *
import torch
import torch.nn as nn
from torchvision import transforms, models
#from models.resnet import *
from sklearn.metrics import f1_score, accuracy_score

if __name__ == '__main__':
    # load sample image
    data_transform_augment = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = Preprocessing("random_test.csv", transform=data_transform_augment)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = 'saved_models/Resnet18_pretrain_cv.pt'
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path))
    #model.to(device)
    model.eval()

    criterion = nn.BCELoss()
    truth = []
    pred = []

    for batch_index, sample in enumerate(dataloader):

        inputs = sample[0]
        gt = sample[1]

        #gt = gt.to(device)

        image = inputs.type(torch.FloatTensor)
        #image = image.to(device)

        outputs = torch.sigmoid(model(image))

        outputs = torch.where(outputs> 0.5, 1, 0)

        truth.append(gt)
        pred.append(outputs)
        #print(gt, outputs)

    print(f1_score(truth, pred), "F1score")

    print(accuracy_score(truth, pred), "Accuracy")

