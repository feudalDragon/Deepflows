from DF_serving.serving.serving import MyDLPWServicer


if __name__ == '__main__':
    host = 'localhost:5000'
    root_dir = 'D:/python_project/onnx_file'
    model_file_name = 'CNN_MNIST.onnx'
    serving = MyDLPWServicer(host, root_dir, model_file_name)
    serving.serve()
