from torch.utils.data import Dataset
import os
import numpy as np
import random



class EmbeddingsDataset(Dataset):
    def __init__(self, folder_path, df, subset=None):
        self.folder_path = folder_path
        self.df = df
        self.subsample = subset

        if(self.subsample == None):
            
            self.files_list = os.listdir(self.folder_path)
        else:
            print("Randomly selecting {} embeddings".format(self.subsample))
            self.files_list = random.sample(os.listdir(self.folder_path), self.subsample)

        #self.num_splits = len(os.listdir(self.folder_path))
        #self.df_splits = np.array_split(self.df, self.num_splits)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        embedding = np.load(os.path.join(self.folder_path, self.files_list[idx]))
        filename = self.files_list[idx].strip('.npy')+'.jpeg'
        #print("$$$$$$$$$$$$$$$" ,type(self.df[self.df['image'] == filename]['level']))
        label = self.df[self.df['image'] == filename]['level'].to_numpy()

        #labels = np.array(self.df['level'][idx])
        # print(embedding)
        #print("labels from within dataloader", label)
        # print(filename)
        return embedding, label, filename


    