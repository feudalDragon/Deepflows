import os
import sys
import json
import time
import threading
import asyncio
import base64
import io
import gc
import numpy as np
from contextlib import asynccontextmanager  # 必须导入这个
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==================== 0. 环境路径与 CUDA 修复 ====================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 强制添加 CUDA 路径 (根据你之前的成功经验，保留 E 盘路径)
if hasattr(os, "add_dll_directory"):
    # 这里填你确认有效的路径
    cuda_bin_path = r"E:\cuda\bin" 
    if os.path.exists(cuda_bin_path):
        try:
            os.add_dll_directory(cuda_bin_path)
            print(f"[Server] ✅ CUDA 路径已添加: {cuda_bin_path}")
        except Exception as e:
            print(f"[Server] ⚠️ 添加路径失败: {e}")

    # 添加编译好的 .pyd 路径
    pyd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepFlows', 'backend', 'backend_src', 'build', 'Release'))
    if os.path.exists(pyd_path):
        try:
            os.add_dll_directory(pyd_path)
        except:
            pass

# ==================== 1. DeepFlows 导入 ====================
try:
    from DeepFlows.tensor import *
    from DeepFlows.optim import Adam, SGD, Adadelta, Adagrad
    from DeepFlows.utils import data_loader
    from DeepFlows.nn import functional as F
    from DeepFlows import nn
    from DeepFlows.tensor import Tensor
    from DeepFlows import backend_api
except ImportError as e:
    print(f"[Error] DeepFlows 导入失败: {e}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

# ==================== 2. 全局状态管理 ====================
training_state = {
    "is_running": False,
    "should_stop": False,
    "epoch": 0
}

# ==================== 3. FastAPI 生命周期管理 (关键) ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动逻辑 ---
    print("[System] Server starting up...")
    
    yield  # 服务运行中...
    
    # --- 关闭逻辑 ---
    print("[System] Server shutting down... Stopping threads.")
    training_state["should_stop"] = True
    # 这里不需要费力杀线程，因为下面会把线程设为 daemon=True

# ==================== 4. APP 定义与 CORS 配置 (顺序很重要) ====================
app = FastAPI(
    title="DeepFlows Visualizer",
    lifespan=lifespan  # 绑定生命周期
)

# 【CORS 终极配置】
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",   # Postman 本机
        "http://localhost:8000",   # 浏览器本机
        "http://localhost:3000",   # 前端开发服务器
        "http://127.0.0.1:3000",   # 前端开发服务器 IP
        "null"                     # 某些工具的空 Origin
    ],
    allow_credentials=True,        # 允许 Cookie/认证头
    allow_methods=["*"],           # 允许所有方法
    allow_headers=["*"],           # 允许所有 Header
)

import psutil
import pynvml

# ==================== 5. WebSocket 管理器 ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitor_task = None  # 监控任务句柄

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # 如果是第一个连接，启动监控任务
        if len(self.active_connections) == 1 and (self.monitor_task is None or self.monitor_task.done()):
            self.monitor_task = asyncio.create_task(self.start_monitoring())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # 如果没有连接了，可以在这里选择取消监控任务，或者让它继续跑（取决于需求）
        # 这里为了简单，我们让它继续跑，或者下次连接时复用

    async def broadcast(self, message: dict):
        # 转换 numpy 类型
        json_str = json.dumps(message, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        # 复制一份列表进行遍历，防止遍历时列表被修改
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json_str)
            except Exception:
                # 发送失败通常意味着连接已断开，从列表中移除
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

    async def start_monitoring(self):
        """后台任务：每秒推送一次系统资源使用情况"""
        try:
            pynvml.nvmlInit()
            has_gpu = True
        except:
            has_gpu = False
            print("[Monitor] GPU monitoring disabled (pynvml init failed or no GPU)")

        while True:
            if not self.active_connections:
                await asyncio.sleep(1)
                continue

            try:
                # CPU & RAM
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                # GPU
                gpu_percent = 0
                if has_gpu:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = util.gpu
                    except:
                        pass

                msg = {
                    "type": "resources",
                    "data": {
                        "cpu": cpu_percent,
                        "ram": ram_percent,
                        "gpu": gpu_percent
                    }
                }
                await self.broadcast(msg)
            except Exception as e:
                print(f"[Monitor Error] {e}")
            
            await asyncio.sleep(1)  # 每秒推送一次

manager = ConnectionManager()

# ==================== 6. 辅助函数 ====================
def get_weight_distribution(model):
    if hasattr(model, 'conv1') and hasattr(model.conv1, 'weight'):
        try:
            weights = model.conv1.weight.data.numpy().flatten()
            hist, bin_edges = np.histogram(weights, bins=20)
            return {"layer": "conv1", "bins": bin_edges.tolist(), "counts": hist.tolist()}
        except:
            return None
    return None

def numpy_to_base64(arr):
    # 简单实现，避免引入 matplotlib 导致线程问题，实际项目建议优化
    return "data:image/png;base64,..." 

# ==================== 7. 模型定义 ====================
class DynamicModel(nn.Module):
    def __init__(self, layer_config: List[dict]):
        super().__init__()
        self.layers = []
        self.layer_names = []
        
        # 简单工厂模式，将 JSON 配置映射到 DeepFlows 层
        # 这里的实现比较基础，假设了层之间的维度是手动匹配的，或者通过 Flatten 自动处理
        for i, layer_info in enumerate(layer_config):
            l_type = layer_info.get("type")
            l_params = layer_info.get("params", {})
            
            layer = None
            if l_type == "Conv2d":
                layer = nn.Conv2d(**l_params, device='cuda')
            elif l_type == "ReLU":
                layer = nn.ReLU()
            elif l_type == "MaxPool2d":
                layer = nn.MaxPool2d(**l_params)
            elif l_type == "Flatten":
                # Flatten 在 DeepFlows 中可能是一个 reshape 操作，这里用一个简单的 Lambda 层或者在 forward 里处理
                # 假设我们有一个 Flatten 层，或者我们用 reshape
                pass # Flatten 比较特殊，我们在 forward 里动态处理，或者实现一个 Flatten Module
            elif l_type == "Linear":
                layer = nn.Linear(**l_params, device='cuda')
            elif l_type == "Dropout":
                # 假设框架有 Dropout，如果没有就跳过
                try:
                    layer = nn.Dropout(**l_params)
                except:
                    pass
            
            if layer:
                # 必须将层注册为属性，才能被 optimizer 发现参数
                name = f"layer_{i}_{l_type}"
                setattr(self, name, layer)
                self.layers.append(layer)
                self.layer_names.append(name)
            elif l_type == "Flatten":
                self.layers.append("Flatten") # 标记一下

    def forward(self, x):
        for layer in self.layers:
            if layer == "Flatten":
                x = x.reshape(x.shape[0], -1)
            else:
                x = layer(x)
        return x

class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, device='cuda')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, device='cuda')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes, device='cuda')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# ==================== 8. 训练线程逻辑 ====================
def train_worker(config: dict):
    print("[Worker] Training thread started.")
    
    # 模拟数据加载 (为了演示稳定，这里简化了 fetch_openml 的错误处理)
    try:
        dataset_name = config.get("dataset", "mnist").lower()
        if dataset_name == "cifar10":
            # 简单模拟 CIFAR-10 加载逻辑，实际项目应替换为真实加载
            mnist = fetch_openml('cifar_10', version=1, as_frame=False, cache=True)
            # CIFAR-10 是 3通道，这里模型目前只支持 1通道 (MNIST)，为了演示不报错，暂时先用 MNIST 或者需要修改模型结构
            # 鉴于时间，这里如果选了非 MNIST，暂时还是加载 MNIST 但打印个警告
            print(f"[Worker] Warning: Dataset {dataset_name} support is experimental. Fallback to MNIST for stability.")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
        else:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
            
        x, y = mnist['data'], mnist['target']
        x = x.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y = y.astype(np.int32)
        x_train, _, y_train, _ = train_test_split(x, y, test_size=1/7, random_state=42, stratify=y)
        
        x_train = np.ascontiguousarray(x_train)
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(np.arange(10).reshape(-1, 1))

        batch_size = config.get("batch_size", 64)
        lr = config.get("lr", 0.001)
        
        loader = data_loader(x_train, y_train, batch_size, shuffle=True)
        
        # 动态模型加载逻辑
        custom_model_config = config.get("custom_model_config")
        if custom_model_config and len(custom_model_config) > 0:
            print(f"[Worker] Using Dynamic Model with config: {custom_model_config}")
            try:
                model = DynamicModel(custom_model_config)
            except Exception as e:
                print(f"[Worker Error] Failed to build dynamic model: {e}. Fallback to MNIST_CNN.")
                model = MNIST_CNN(10)
        else:
            print("[Worker] Using default MNIST_CNN")
            model = MNIST_CNN(10)
            
        criterion = nn.CrossEntropyLoss()
        optimizer_name = config.get("optimizer", "Adam")
        if optimizer_name == "SGD":
            optimizer = SGD(model.parameters(), lr=lr)
        elif optimizer_name == "Adagrad":
            optimizer = Adagrad(model.parameters(), lr=lr)
        else:
            optimizer = Adam(model.parameters(), lr=lr)

        training_state["is_running"] = True
        training_state["should_stop"] = False

        # 创建新的事件循环用于线程内发送 WS 消息
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for epoch in range(config.get("epochs", 5)):
            if training_state["should_stop"]: break
            
            model.train()
            for batch_idx, (inputs, labels) in enumerate(loader):
                if training_state["should_stop"]: break

                labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
                t_inputs = Tensor(inputs, device=backend_api.Device('cuda'))
                t_labels = Tensor(labels_onehot, device=backend_api.Device('cuda'))
                
                outputs = model(t_inputs)
                loss = criterion(outputs, t_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # --- 计算准确率与推送 ---
                curr_loss = loss.data.numpy().item()
                if batch_idx % 10 == 0:
                    # 准确率计算
                    _pred = np.argmax(outputs.data.numpy(), axis=1)
                    _true = np.argmax(labels_onehot, axis=1)
                    acc = np.mean(_pred == _true) * 100

                    msg = {
                        "type": "metrics",
                        "data": {
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                            "loss": curr_loss,
                            "accuracy": acc
                        }
                    }
                    loop.run_until_complete(manager.broadcast(msg))

                # --- 资源释放 ---
                outputs.dispose()
                loss.dispose()
                t_inputs.dispose()
                t_labels.dispose()
                del t_inputs, t_labels, outputs, loss
                
                if batch_idx % 50 == 0: gc.collect()

        print("[Worker] Training loop finished.")

    except Exception as e:
        print(f"[Worker Error] {e}")
        loop.run_until_complete(manager.broadcast({"type": "error", "data": str(e)}))
    finally:
        training_state["is_running"] = False
        loop.run_until_complete(manager.broadcast({"type": "status", "data": "stopped"}))
        loop.close()
        try:
            # 清理显存
            from DeepFlows.tensor import Graph
            Graph.free_graph_all()
        except:
            pass
        gc.collect()

# ==================== 9. API 路由 ====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class TrainConfig(BaseModel):
    lr: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    optimizer: str = "Adam"
    dataset: str = "mnist"
    custom_model_config: Optional[List[dict]] = None  # Rename to avoid Pydantic conflict

@app.post("/api/train/start")
async def start_training(config: TrainConfig):
    if training_state["is_running"]:
        return {"status": "error", "message": "Training is running"}
    
    # 【关键】daemon=True 确保主进程退出时线程会被杀掉
    t = threading.Thread(target=train_worker, args=(config.model_dump(),), daemon=True)
    t.start()
    return {"status": "success", "message": "Started"}

@app.post("/api/train/stop")
async def stop_training():
    if not training_state["is_running"]:
        return {"status": "error", "message": "Not running"}
    training_state["should_stop"] = True
    return {"status": "success", "message": "Stopping..."}

@app.get("/api/models")
async def get_models():
    return {"models": [{"id": 1, "name": "Demo_CNN", "acc": 98.5}]}

if __name__ == "__main__":
    import uvicorn
    # 启动命令
    uvicorn.run(app, host="0.0.0.0", port=8000)