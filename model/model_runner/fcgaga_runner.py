import torch

from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape


class FcGagaRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
        self.forward_features = cfg["MODEL"].get("FROWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data = data
        history_data        = self.to_running_device(history_data)      # B, L, N, C
        future_data         = self.to_running_device(future_data)       # B, L, N, C

        history_data = history_data.transpose(1, 2) # B, N, L, C
        future_data = self.select_target_features(future_data)

        history_in = history_data[..., 0]
        time_of_day_in = history_data[..., 1, 1]

        B, N, L, C = history_data.shape
        node_id_in = self.to_running_device(torch.arange(N).unsqueeze(0).repeat(B, 1))
        # feed forward history_in, node_id_in, time_of_day_in
        prediction = self.model(history_in=history_in, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
        prediction = prediction.unsqueeze(-1).transpose(1, 2)
        batch_size, length, num_nodes, _ = future_data.shape

        return prediction, future_data
