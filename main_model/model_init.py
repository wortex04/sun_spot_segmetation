
from main_model.u_net_model import U_Net
import torch
def model_initification(path='main_model/w4_io_0.0619078509885223_loss_0.1604776674260696data2024-04-09 12_17_13.325687.pth'):
    model = U_Net()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
model_initification()
