import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from dataset import embeddings_dataset
from torch.utils.data import DataLoader
from model import logReg
from time import time


root_path = '/home/co-dutt1/rds/hpc-work/SSL-Augmentations/Embeddings'
model = 'dorsal'
dataset = 'retinopathy'
create_embeddings_for = 'train'
train_df = pd.read_csv('/home/co-dutt1/rds/hpc-work/MLP-RIQA/Train_set_RIQA_DR_Labels.csv')
test_df = pd.read_csv('/home/co-dutt1/rds/hpc-work/MLP-RIQA/Test_set_RIQA_DR_Labels.csv')

BATCH_SIZE = 1
train_folder_path = os.path.join(root_path, dataset, 'train', model)
test_folder_path = os.path.join(root_path, dataset, 'test', model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = embeddings_dataset.EmbeddingsDataset(train_folder_path, train_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

#test_dataset = embeddings_dataset.EmbeddingsDataset(test_folder_path, test_df)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(len(train_dataset))
#print(len(test_dataset))

lr_rate = 0.0001
n_iters = 3000
epochs = 1
NUM_CLASSES = 5
EMBEDDINGS_DIM = 2048

logreg_model = logReg.LogisticRegression(EMBEDDINGS_DIM, NUM_CLASSES).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(logreg_model.parameters(), lr=lr_rate)

iter = 0

for epoch in range(int(epochs)):
    start_time = time()
    print('\n')
    print("EPOCH {} out of {}".format(epoch+1, epochs))
    for i, (embedding, labels, filename) in enumerate(train_loader):
        #print(embedding.shape, labels.shape)

        embedding, labels = embedding.to(device).squeeze(1), labels.to(device)
        #print(labels, filename)
        print("Embedding shape", embedding.shape)
    
        optimizer.zero_grad()

        outputs = logreg_model(embedding)
        # print("Outputs shape: ", outputs.shape)
        print("Labels",labels, labels.shape, labels.dtype)

        print("outputs", outputs, outputs.shape, outputs.dtype)
        #print("argmax outputs", torch.argmax(outputs, dim=1), torch.argmax(outputs, dim=1).shape, torch.argmax(outputs, dim=1).dtype)
        # print("labels", labels[0])  

        loss = criterion(outputs, torch.squeeze(labels, dim=1))

        loss.backward()

        optimizer.step()

        if(i % 5 == 0):
            print("Train Loss: {}".format(loss))
            
            correct = 0
            total = 0

    # for j, (test_embd, test_labels, filename) in enumerate(test_loader):
    #     test_embd, test_labels = test_embd.to(device).squeeze(1), test_labels.to(device)

    #     test_output = logreg_model(test_embd)
    #     _, predicted = torch.max(test_output, dim=1)
    #     # print("filename: ", filename)
    #     # print("Outputs shape: ", test_output.shape)
    #     # print("Labels shape: ", test_labels.shape)

    #     test_loss = criterion(test_output, torch.squeeze(test_labels, dim=1))
    #     total += test_labels.detach().size(0)
    #     print("total: ", total)

    #     #print(test_output.shape, predicted.shape, test_labels.shape)
    #     #print(predicted)
    #     #print(test_labels)

    #     correct += (predicted == test_labels).sum()
    
    # accuracy = 100*correct/len(test_loader)
    # end_time = time()
    # seconds_elapsed = end_time - start_time
    # hours, rest = divmod(seconds_elapsed, 3600)
    # minutes, seconds = divmod(rest, 60)

    # print("Iteration: {} | Test Loss: {} | Accuracy: {}".format(iter, test_loss.item(), accuracy))
    # print("This epoch took {} minutes and {} seconds".format(minutes, seconds))
