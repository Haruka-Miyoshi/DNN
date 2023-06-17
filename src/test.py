from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dnn import DNN

# 検証用データ
test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# ミニバッチのサイズ
batch_size=64

# 検証用データの読み込み
test_dataloader=DataLoader(test_data, batch_size, shuffle=True)

# DNN インスタンス生成
model=DNN()

# テスト
model.test_accuracy(test_dataloader, mode=True)