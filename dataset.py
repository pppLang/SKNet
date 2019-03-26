import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, mode='train', root_path='D:\\data_cifia10\\'):
        super(MyDataset, self).__init__()
        if mode == 'train':
            file_path = os.path.join(root_path, 'data_batch_{}')
            self.data, self.labels = load_traindata(file_path=file_path)
        elif mode == 'test':
            file_path = os.path.join(root_path, 'test_batch')
            data_dict = unpickle(file_path)
            self.data = data_dict[b'data']
            self.labels = data_dict[b'labels']
        self.data = self.data/255
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index, :].reshape(3, 32, 32).astype(np.float32), self.labels[index]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_traindata(file_path):
    train_data = None
    train_labels = None
    if not os.path.exists(file_path.format(1)):
        print('wrong dataset path : {}'.format(file_path.format(1)))
        exit()
    for i in range(5):
        data_dict = unpickle(file_path.format(i+1))
        if train_data is None:
            train_data = data_dict[b'data']
            train_labels = data_dict[b'labels']
        else:
            train_data = np.concatenate((train_data, data_dict[b'data']), axis=0)
            train_labels = np.concatenate((train_labels, data_dict[b'labels']), axis=0)
    # print(train_data.shape, train_labels.shape)
    return train_data, train_labels



if __name__=="__main__":
    dataset = MyDataset()
    print(len(dataset))
    data, label = dataset[0]
    print(data.shape)
    print(data.max(), data.min())
    print(label)
    plt.figure()
    plt.imshow(np.transpose(data, [1,2,0]))
    plt.show()