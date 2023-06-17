from torch import nn

"""Model Define"""
class Model(nn.Module):
    def __init__(self):
        """initialize"""
        super(Model, self).__init__()
        # 次元圧縮
        self.flatten=nn.Flatten()
        # 全結合層 定義
        self.network=nn.Sequential(
            # 画像サイズ28×28=784次元を512次元へ線形変換
            nn.Linear(28*28, 512),
            # 活性化関数 負値を除去
            nn.ReLU(),
            # 512次元を512次元に変換
            nn.Linear(512, 512),
            # 活性化関数 負値を除去
            nn.ReLU(),
            # 512次元を10次元に変換
            nn.Linear(512, 10),
        )
    
    """forward:順伝播"""
    def forward(self, x):
        # 次元圧縮
        x=self.flatten(x)
        # DNNへ入力
        y=self.network(x)
        # 10次元のデータを出力
        return y