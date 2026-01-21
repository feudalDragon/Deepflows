import random
import grpc
import numpy as np
# from DF_serving.serving.proto import serving_pb2, serving_pb2_grpc

import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set()

# 数据预处理：独热化+标准化
data_x, data_y = fetch_openml('mnist_784', version=1, return_X_y=True)
data_x, data_y = data_x[:5000].values, data_y[:5000].values
data_y = OneHotEncoder(sparse=False).fit_transform(data_y.reshape(-1, 1))
x_train, y_train, x_test, y_test = train_test_split(
    data_x,
    data_y,
    train_size=0.7,
)
stder = StandardScaler()
x_train = stder.fit_transform(x_train).reshape(-1, 1, 28, 28)
y_train = stder.transform(y_train).reshape(-1, 1, 28, 28)


class MyDLPWServingClient(object):
    def __init__(self, host):
        self.stub = serving_pb2_grpc.MyDLPWServingStub(
            grpc.insecure_channel(host))

    def Predict(self, value_tobe_pre):
        req = serving_pb2.PredictReq()
        for value in value_tobe_pre:
            proto_val = req.data.add()
            proto_val.value.extend(np.array(value).flatten())
            proto_val.dim.extend(list(value.shape))
        resp = self.stub.Predict(req)
        return resp


if __name__ == '__main__':
    host = 'localhost:5000'
    # client = MyDLPWServingClient(host)

    while True:
        rand_index = random.randint(0, len(x_train))
        img = x_train[rand_index]
        label = y_train[rand_index]
        print(img.shape)
        print(label.shape)
        break
"""        print('Send prediction request...')
        resp_mat = client.Predict([img])"""
