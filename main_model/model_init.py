from main_model.u_net_model import U_Net
import requests
import torch
import io


def model_initification(path='https://storage.yandexcloud.net/marsel1/w2_io_loss_0.48305392265319824_loss_0.3506365865468979data2024-04-29%2010_34_50.416048.pth'):
    # Send a GET request to download the file
    response = requests.get(path, stream=True)
    # Create a seekable buffer from the response content
    buffer = io.BytesIO(response.content)
    # Load the model weights from the buffer
    model_weights = torch.load(buffer, map_location='cpu')
    model = U_Net()
    model.load_state_dict(model_weights)
    model.eval()
    return model
