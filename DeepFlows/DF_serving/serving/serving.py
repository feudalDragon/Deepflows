import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os

import grpc
import onnxruntime as ort

from DF_serving.serving.proto import serving_pb2, serving_pb2_grpc


class MyDLPWServingService(serving_pb2_grpc.MyDLPWServingServicer):
    r"""
    推理服务，主要流程:

    """

    def __init__(self, root_dir, model_file_name):
        self.root_dir = root_dir
        self.model_file_name = model_file_name

        model_file_path = os.path.join(root_dir, model_file_name)
        self.ort_sess = ort.InferenceSession(model_file_path)

    def Predict(self, predict_req, context):

        # 从protobuf数据反序列化成模型输入数据格式
        data_tobe_inference = MyDLPWServingService.deserialize(predict_req)

        # 调用模型得到输出，即预测结果
        inference_result = self._inference(data_tobe_inference)

        # 预测结果序列化为protobuf格式，通过网络返回
        predict_resp_proto = MyDLPWServingService.serialize(inference_result)

        return predict_resp_proto

    @staticmethod
    def deserialize(predict_req):
        infer_req_tensor_list = []
        for proto_tensor in predict_req.data:
            dim = tuple(proto_tensor.dim)
            np_tensor = np.array(proto_tensor.value, dtype=np.float32)
            np_tensor = np.reshape(np_tensor, dim)
            infer_req_tensor_list.append(np_tensor)

        return infer_req_tensor_list

    @staticmethod
    def serialize(inference_resp):
        resp = serving_pb2.PredictResp()
        for np_tensor in inference_resp:
            proto_tensor = resp.data.add()
            proto_tensor.value.extend(np.array(np_tensor).flatten())
            proto_tensor.dim.extend(list(np_tensor.shape))

        return resp

    def _inference(self, inference_req):
        inference_resp_tensor_list = []

        for np_tensor in inference_req:
            output = self.ort_sess.run(None, {'inputs': np_tensor})
            inference_resp_tensor_list.append(output)

        return inference_resp_tensor_list


class MyDLPWServicer(object):
    def __init__(self, host, root_dir, model_file_name, max_workers=10):
        self.host = host
        self.max_workers = max_workers

        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))

        serving_pb2_grpc.add_MyDLPWServingServicer_to_server(
            MyDLPWServingService(root_dir, model_file_name), self.server)

        self.server.add_insecure_port(self.host)

    def serve(self):

        # 启动rpc服务
        self.server.start()

        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)
