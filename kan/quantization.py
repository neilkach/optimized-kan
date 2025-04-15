import torch.quantization

class QuantizedKAN(torch.nn.Module):
    def __init__(self, kan_model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.kan = kan_model

    def forward(self, x):
        x = self.quant(x)
        x = self.kan(x)
        return self.dequant(x)

# Post-training quantization
def quantize_model(model, dtype=torch.qint8):
    quantized_model = QuantizedKAN(model)
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    return torch.quantization.prepare(quantized_model, inplace=True)
