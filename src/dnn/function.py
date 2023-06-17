import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from .model import Model

"""Deep Neural Network"""
class DNN(object):
    """initialize"""
    def __init__(self, mode=False, model_path='') -> None:
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル定義
        self.__model=Model().to(self.__device)

        # 学習済みモデル
        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))

        # 学習係数
        self.__lr=1e-3
        # 損失関数:交差エントロピー
        self.__loss_func=nn.CrossEntropyLoss()
        # 最適化アルゴリズム:SGD
        self.__opt=torch.optim.SGD(self.__model.parameters(), lr=self.__lr)

        # save file path
        self.FILE_PATH=os.path.join('./model')

        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)

        # 損失値格納用変数
        self.__loss_history=[]

    """update:学習"""
    def update(self, data, mode=False):
        data=tqdm(data)
        # パラメータ計算
        for batch, (X, y) in enumerate(data):
            # device調整
            X=X.to(self.__device)
            y=y.to(self.__device)
            # 学習用データXをDNNモデルに入力 -> 計算結果 出力Y
            pred=self.__model(X)
            # 損失計算(ラベルYと予測Yとの交差エントロピーを計算)
            loss=self.__loss_func(pred, y)

            # 誤差逆伝播を計算
            # 勾配値を0にする
            self.__opt.zero_grad()
            # 逆伝播を計算
            loss.backward()
            # 勾配を計算
            self.__opt.step()
            
            loss=loss.item()
            # 損失を格納
            self.__loss_history.append(loss)

        # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'loss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, self.__loss_history)
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'parameter.txt')
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
        
        # 完了表示
        print("\n ====================== \n")
        print(" ====== complete ====== ")
        print("\n ====================== \n")

    """test_accuracy:テストデータを使った精度評価"""
    def test_accuracy(self, data, mode=False):
        data=tqdm(data)
        # 勾配なし
        with torch.no_grad():
            # 汎用的なデータセットに対応
            n=0
            # 損失
            loss=0
            # 精度
            acc=0
            # 精度
            correct=0
            # パラメータ計算
            for batch, (X, y) in enumerate(data):
                # device調整
                X=X.to(self.__device)
                y=y.to(self.__device)
                # 予測
                pred=self.__model(X)
                # 損失計算
                loss+=self.__loss_func(pred, y).item()
                # 精度計算
                correct+=(pred.argmax(1) == y).type(torch.float).sum().item()
                # データ数 計算
                n+=1
            
            # 精度[%]
            acc=100*(correct/(n+1))
        
        print("\n ====================== \n")
        print(f"loss:{loss}, acc:{acc}")
        print("\n ====================== \n")

                # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'testloss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, [loss])
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'acc.txt')
            # 学習したパラメータを保存
            np.savetxt(PARAM_SAVE, [acc])

        return loss, acc
    
    """prediction:予測"""
    def prediction(self, X):
        X=X.to(self.__device)
        # 予測
        pred=self.__model(X)
        
        print("\n ====================== \n")
        print(f"y:{pred}")
        print("\n ====================== \n")     

        return pred