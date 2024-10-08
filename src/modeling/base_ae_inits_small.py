import torch
import torch.nn as nn

from src.modeling.base import BaseModel


class BaseAutoEncoderWithInitsSmall(BaseModel):
    def __init__(self, model_name='base_auto_encoder', init_method=None):
        super(BaseAutoEncoderWithInitsSmall, self).__init__(model_name)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

        self.initialize_weights(init_method)

    def initialize_weights(self, init_method):
        def init_func(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(init_func)

        
    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x