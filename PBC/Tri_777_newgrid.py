import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import os
import numpy as np

# # 결과를 저장할 디렉토리 지정 : class 호출시 앞에서 선언
# save_directory = "Trigono_7,7,7_MSE-1"
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)


class PINNModel:
    def __init__(self, A, B, C, epochs, model_path, save_directory, device='cpu'):
        self.A = A
        self.B = B
        self.C = C
        self.epochs = epochs
        self.device = device
        # Load or build the model accordingly
        if model_path:
            self.model = self._build_model(model_path)
        else:
            self.model = self._build_model_first()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=200, verbose=True)
        self.criterion = nn.MSELoss()
        self.loader = self._create_dataloader()
        self._initialize_tracking_variables()
        self.loss_history = []  # loss_history를 초기화합니다.
        self.data_loss_history = []
        self.pde_loss_history = []
        # self.initial_total_loss = []
        self.save_directory = save_directory
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        
        
        # 훈련 데이터 준비
        x_data, y_data = np.meshgrid(
            np.linspace(-30, 30, 100),
            np.linspace(-30, 30, 100)
        )
        f_data = A * np.sin(x_data/B) * np.sin(y_data/C)
        xy_data = np.stack([x_data.ravel(), y_data.ravel()], axis=-1)
        self.xy_data = torch.tensor(xy_data, dtype=torch.float32)
        self.f_data = torch.tensor(f_data.ravel(), dtype=torch.float32).view(-1, 1)


        # 테스트 데이터 준비
        x_test, y_test = np.meshgrid(
            np.linspace(-30, 30, 100),
            np.linspace(-30, 30, 100)
        )
        f_test = A * np.sin(x_test/B) * np.sin(y_test/C)
        xy_test = np.stack([x_test.ravel(), y_test.ravel()], axis=-1)
        self.x_test = x_test  # NumPy 배열 (그래프 작성용)
        self.y_test = y_test  # NumPy 배열 (그래프 작성용)
        self.xy_test_tensor = torch.tensor(xy_test, dtype=torch.float32, device=device)
        self.f_test = torch.tensor(f_test.ravel(), dtype=torch.float32, device=device).view(-1, 1)  # 텐서로 변경




    class Net_PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 10)
            self.fc4 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = self.fc4(x)
            return x


    # def _build_model(self, model_path):
    #     model = self.Net_PINN()
    #     # DataParallel로 모델을 래핑합니다.
    #     model = nn.DataParallel(model)
    #     # 저장된 상태 사전을 불러옵니다.
    #     model.load_state_dict(torch.load(model_path))
    #     return model

    def _build_model(self, model_path):
        model = self.Net_PINN()
        # 저장된 상태 사전을 불러옵니다.
        state_dict = torch.load(model_path)
        # 'module.' 접두사를 제거하여 새로운 상태 사전을 생성합니다.
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model
    
    def _build_model_first(self):
        return self.Net_PINN()
    

    def _create_dataloader(self):
        x_data, y_data = np.meshgrid(np.linspace(-30, 30, 50), np.linspace(-30, 30, 50))
        f_data = self.A * np.sin(x_data/self.B) * np.sin(y_data/self.C)
        xy_data = np.stack([x_data.ravel(), y_data.ravel()], axis=-1)
        xy_data = torch.tensor(xy_data, dtype=torch.float32)
        f_data = torch.tensor(f_data.ravel(), dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(xy_data, f_data)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        return loader


    def compute_derivatives(self, model, xy_data):
        xy_data = xy_data.to(self.device)
        # Ensure that xy_data has gradient information.
        xy_data.requires_grad_(True)     
        # Get the model prediction.
        f_pred = model(xy_data)        
        # Create a tensor of ones with the same shape as f_pred to be used for gradient computation.
        # Reshape the ones tensor to match the shape of f_pred.
        ones = torch.ones(f_pred.shape, device=self.device, requires_grad=False)        
        # Compute the first derivatives.
        f_x = torch.autograd.grad(f_pred, xy_data, grad_outputs=ones, create_graph=True)[0][:, 0]
        f_y = torch.autograd.grad(f_pred, xy_data, grad_outputs=ones, create_graph=True)[0][:, 1]        
        # Compute the second derivatives.
        f_xx = torch.autograd.grad(f_x, xy_data, grad_outputs=ones[:, 0], create_graph=True)[0][:, 0]
        f_yy = torch.autograd.grad(f_y, xy_data, grad_outputs=ones[:, 0], create_graph=True)[0][:, 1]       
        return f_xx, f_yy


    def _initialize_tracking_variables(self):
        self.grad_changes = {}
        self.param_values_at_epoch_100 = {}
        self.weight_magnitudes = {}
        for layer_num, (name, param) in enumerate(self.get_model().named_parameters()):
            if "bias" not in name:
                for idx in range(param.numel()):
                    param_id = f"L{layer_num+1}_{idx}"
                    self.grad_changes[param_id] = []
                    self.weight_magnitudes[param_id] = []
                    
    def get_model(self):
        # This function ensures that we return the correct model
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model
        
    def save_prediction_with_mse(self, epoch, x_test, y_test, xy_test_tensor):
        self.model.eval()
        xy_test_tensor = xy_test_tensor.to(self.device)  # GPU로 이동
        with torch.no_grad():
            f_pred = self.model(xy_test_tensor).cpu().numpy().reshape(100, 100)  # 예측 결과를 CPU로 이동 후 NumPy 배열로 변환
            mse = np.mean((f_pred - self.f_test.cpu().numpy().reshape(100, 100))**2)  # MSE 계산

        fig, ax = plt.subplots()
        cp = ax.contourf(x_test, y_test, f_pred, cmap='seismic')
        fig.colorbar(cp, ax=ax)
        ax.set_title(f'{epoch} epoch prediction, MSE: {mse:.5f}')
        plt.savefig(os.path.join(self.save_directory, f'Prediction-{epoch}epoch.png'))
        plt.close()


    # 모델 테스트를 위한 함수를 정의합니다.
    def test_model(self, xy_test_tensor, f_test):
        self.model.eval()
        xy_test_tensor = xy_test_tensor.to(self.device)  # GPU로 이동
        f_test = f_test.to(self.device)  # GPU로 이동
        with torch.no_grad():
            predictions = self.model(xy_test_tensor).cpu().numpy()
            predictions = predictions.reshape(f_test.cpu().numpy().shape)
        mse = np.mean((predictions - f_test.cpu().numpy())**2)
        return mse


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            batch_loss = 0.0
            batch_data_loss = 0.0
            batch_pde_loss = 0.0
            
            for batch_num, (batch_xy, batch_f) in enumerate(self.loader):
                batch_xy, batch_f = batch_xy.to(self.device), batch_f.to(self.device)
                self.optimizer.zero_grad()
                
                f_pred = self.model(batch_xy)
                data_loss = self.criterion(f_pred, batch_f)
                f_xx, f_yy = self.compute_derivatives(self.model, batch_xy)
                pde_loss = self.criterion(f_xx + f_yy, ((1/self.B**2)+(1/self.C**2))*(-1)*f_pred.squeeze())
                
                loss = data_loss + pde_loss
                loss.backward()
                self.optimizer.step()
                
                batch_loss += loss.item()
                batch_data_loss += data_loss.item()
                batch_pde_loss += pde_loss.item()

            avg_loss = batch_loss / len(self.loader)
            avg_data_loss = batch_data_loss / len(self.loader)
            avg_pde_loss = batch_pde_loss / len(self.loader)
            
            self.loss_history.append(avg_loss)
            self.data_loss_history.append(avg_data_loss)
            self.pde_loss_history.append(avg_pde_loss)
            
            self.scheduler.step(avg_loss)
            
            if (epoch < 10) or (epoch % 100 == 0):
                self.save_prediction_with_mse(epoch, self.x_test, self.y_test, self.xy_test_tensor)
            
            if epoch == 50 or epoch == 100:
                mse = self.test_model(self.xy_test_tensor, self.f_test)
                print(f'Epoch {epoch}, MSE: {mse}')


# # 사용 예시
# # model_path = '777_1.pth'  # 모델 파일 경로 지정
# A_value = 5
# B_value = 7
# C_value = 7
# save_directory = f"Trigono_{A_value},{B_value},{C_value}_MSE-1"

# pinn = PINNModel(A=A_value, B=B_value, C=C_value, epochs=5000, model_path="initial.pth", save_directory=save_directory)
# pinn.train()
