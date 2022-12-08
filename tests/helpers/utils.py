from pytorch_lightning import LightningModule


def count_trainable_params(model: LightningModule) -> int:
    """
    Returns the number of trainable parameters in a model
    Args:
        model (LightningModule): The model object

    Returns:
        (int): The number of trainable parameters

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
