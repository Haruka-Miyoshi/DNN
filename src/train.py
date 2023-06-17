from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dnn import DNN

# 訓練用データ
train_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# ミニバッチのサイズ
batch_size=64

# 訓練用データの読み込み
train_dataloader=DataLoader(train_data, batch_size, shuffle=True)

# DNN インスタンス生成
model=DNN()

# 学習
model.update(train_dataloader, mode=True)