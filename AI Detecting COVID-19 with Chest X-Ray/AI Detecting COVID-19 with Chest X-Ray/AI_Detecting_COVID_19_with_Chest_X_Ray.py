#Dataset from COVID-19 Radiography Dataset on Kaggle

#start imported libraries
import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)
print('Using PyTorch version', torch.__version__)

#end imported libraries

#star preping data sets
class_names = ['normal', 'viral', 'covid']
root_dir = 'COVID-19 Radiography Database'
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 30)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)
class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print('Found {len(images)} {class_name} examples')
            return images
        #creation of a dictionary to keep track of the images outputed in the above function
        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    #returns lenght data set of the three classes combined
    
    def __getitem__(self, index):
        #chooses random class
        class_name = random.choice(self.class_names)
        #there to "fix" the imbalance in inputed training sets as there's less covid images than there is normal ones
        #that way the index actually correspond to a real index in the class
        index = index % len(self.images[class_name])
        #list images that belong to the class
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        #to load
        #pytorch requires transform
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

    #image transformation parameters for training data set
    train_transform = torchvision.transforms.Compose([
    #resizing images
    torchvision.transforms.Resize(size=(227, 227)),
    #flipping images at random to do an augmentation of the training set
    torchvision.transforms.RandomHorizontalFlip(),
    #can be used by pytorch
    torchvision.transforms.ToTensor(),
    #normalize data
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    #same image tranformation parameters for control data set
    control_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#end preping data sets

#start data loaders
train_dirs = {
    'normal': 'COVID-19 Radiography Database/normal',
    'viral': 'COVID-19 Radiography Database/viral',
    'covid': 'COVID-19 Radiography Database/covid'
}

train_dataset = ChestXRayDataset(train_dirs, train_transform)

control_dirs = {
    'normal': 'COVID-19 Radiography Database/test/normal',
    'viral': 'COVID-19 Radiography Database/test/viral',
    'covid': 'COVID-19 Radiography Database/test/covid'
}

control_dataset = ChestXRayDataset(control_dirs, control_transform)

#amount of pictures that will be displayed at a time
batch_size = 6

dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_control = torch.utils.data.DataLoader(control_dataset, batch_size=batch_size, shuffle=True)

#next lines check if the size of the batches is reasonable len(dl_train)+len(dl_control)= total len and len(dl_train) >> len(dl_control)
print('Number of training batches', len(dl_train))
print('Number of test batches', len(dl_control))

#end data loaders

#start data visualization
lass_names = train_dataset.class_names


def show_images(images, labels, preds):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        #batch size
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        #image transformation parameters - pytorch
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        #original values
        image = image * std + mean
        #clip values
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        #prediction will be displayed in green if correct and red if incorrect
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        #x label = control/real diagnose
        plt.xlabel({class_names[int(labels[i].numpy())]}')
        #y label used to see prediction
        plt.ylabel{{class_names[int(preds[i].numpy())]}', color=col)
    #layout of the plot
    plt.tight_layout()
    #display the plot
    plt.show()

#see examples data(ie Xrays)
images, labels = next(iter(dl_train))
show_images(images, labels, labels)

images, labels = next(iter(dl_control))
show_images(images, labels, labels)

#end data visualization

#start torch vision model
#chose resnet so it can run on my pc without too much computational power
#we want to use the pretrained sets thus why we see pretrained
resnet18 = torchvision.models.resnet18(pretrained=True)
#check model architecture
print(resnet18)

#put 3 outpout features, standart is 1000
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
#classification loss function
loss_fn = torch.nn.CrossEntropyLoss()
#we optimize all the parameters of the module
optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

#function to show predictions
def show_preds():
    resnet18.eval()
    #evaluate random images
    images, labels = next(iter(dl_test))
    outputs = resnet18(images)
    #make sure we get the indixes not the values
    #dimension = number of examples -> 1
    _, preds = torch.max(outputs, 1)
    #we want to show the image, prediction and labels
    show_images(images, labels, preds)
#test predictions
show_preds()

#end torch vision model

#start training 
def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        print('='*20)
        print('Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.

        resnet18.train() # set model to training phase

        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 20 == 0:
                print('Evaluating at step', train_step)
                #we want to evaluate accuracy
                acc = 0
                resnet18.eval() # set model to eval phase

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    acc += sum((preds == labels).numpy())

                    val_loss /= (val_step + 1)
                acc = acc/len(control_dataset)
                print('Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}')

                show_preds()

                #train again
                resnet18.train()

                #will train until accuracy reaches or surpasses 0.95
                if accuracy >= 0.95:
                    print('Performance condition satisfied, stopping training.')
                    return
            #training loss
            train_loss /= (train_step + 1)
            print('Training Loss: {train_loss:.4f}')
print('Training complete..')

#low epoch to avoid overusing computational power, higher epoch == higher accuracy
train(epochs=1)

#end training 

#result
show_preds()