import torch


class _ShiftedSoftPlus(torch.nn.Module):
    """
    Shifted softplus as defined in SchNet, NeurIPS 2017.

    :param beta: value for the a more general softplus, default = 1
    :param threshold: values above are linear function, default = 20
    """

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self.register_buffer("log2", torch.log(torch.as_tensor(2.0)))

    def forward(self, x):
        """
        Evaluate shifted softplus

        :param x: torch.Tensor, input
        :return: torch.Tensor, ssp(x)
        """
        return self.softplus(x) - self.log2


ShiftedSoftPlus = _ShiftedSoftPlus()
