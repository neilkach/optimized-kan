import torch.nn.utils.prune as prune
import KANLayer

def prune_kan(model, amount=0.2):
    parameters_to_prune = [
        (layer, 'weight') 
        for layer in model.modules() 
        if isinstance(layer, KANLayer)
    ]
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
