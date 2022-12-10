import torch
import torch.nn as nn
import torch.nn.functional as F


class FcBlock(nn.Module):
    def __init__(self, hidden_units,block_layers, input_size: int, output_size: int, **kw):
        super(FcBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = nn.ModuleList()
        self.block_layers= block_layers
        for i in range(self.block_layers):
            if i == 0:
                self.fc_layers.append(
                    nn.Linear(in_features=input_size,out_features=hidden_units))
            else:
                self.fc_layers.append(
                    nn.Linear(in_features=hidden_units, out_features=hidden_units))
        self.forecast = nn.Linear(in_features=hidden_units, out_features=self.output_size)
        self.backcast = nn.Linear(in_features=hidden_units, out_features=self.input_size)

    def forward(self, inputs):
        h = self.fc_layers[0](inputs)
        h = F.relu(h)
        for i in range(1, self.block_layers):
            h = self.fc_layers[i](h)
            h = F.relu(h)
        backcast = F.relu(inputs - self.backcast(h))
        return backcast, self.forecast(h)


class FcGagaLayer(nn.Module):
    def __init__(self,
                 num_blocks:int,
                 hidden_units,
                 horizon,
                 input_size: int,
                 history_length: int,
                 num_nodes: int,
                 block_layers,
                 epsilon,
                 node_id_dim
                 ):

        super(FcGagaLayer, self).__init__()

        self.num_blocks = num_blocks
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.epsilon = epsilon
        self.horizon = horizon
        self.node_id_dim = node_id_dim
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(FcBlock(hidden_units,
                                       block_layers,
                                       input_size=self.input_size,
                                       output_size=horizon,))

        # node embedding
        self.node_id_em = nn.Embedding(num_embeddings=self.num_nodes,
                                        embedding_dim=self.node_id_dim,)
        nn.init.uniform_(self.node_id_em.weight)
        self.time_gate1 = nn.Linear(in_features=self.node_id_dim+1,
                                    out_features=hidden_units,)

        self.time_gate2 = nn.Linear(in_features=hidden_units,
                                    out_features=horizon)

        self.time_gate3 = nn.Linear(in_features=hidden_units,
                                    out_features=history_length)

    def forward(self, history_in, node_id_in, time_of_day_in, training=False):
        """
        history_in [batch, nodes, length]
        node_id_in [batch, nodes, 1] (0, 1 2 3 ...)
        time_of_day_in [batch, node]
        """
        # batch, nodes, node_id_dim
        node_id = self.node_id_em(node_id_in)
        # drop batch dim,    nodes, node_id_dim
        node_embeddings = node_id[0, :, :].squeeze()
        # 这一步感觉没什么用？
        # node_id = node_id.squeeze(axis=-2)

        # batch, nodes, output_size+1 →batch, nodes, hidden_units
        time_gate = self.time_gate1(torch.concat((node_id, time_of_day_in.unsqueeze(-1)), dim=-1))
        time_gate = F.relu(time_gate)
        # batch, nodes, hidden_units →batch, nodes, hidden_units
        time_gate_forward = self.time_gate2(time_gate)
        time_gate_backward = self.time_gate3(time_gate)

        history_in = history_in / (1.0 + time_gate_backward)
        # E@E^T nodes, nodes
        node_embeddings_dp = node_embeddings@node_embeddings.transpose(0, 1)
        node_embeddings_dp = torch.where(torch.isinf(node_embeddings_dp), torch.zeros_like(node_embeddings_dp), node_embeddings_dp)
        node_embeddings_dp = torch.exp(self.epsilon * node_embeddings_dp)
        # expand dims
        node_embeddings_dp = node_embeddings_dp[None, :, :, None]
        # batch, nodes, length→ batch, nodes
        level, _ = torch.max(history_in, dim=-1, keepdim=True)

        # like a normalize
        # batch, nodes, length
        history = torch.div(history_in, level)
        history = torch.where(torch.isinf(history), torch.zeros_like(history), history)


        shape = history_in.shape
        # batch, nodes, length → batch, nodes, nodes, length
        all_node_history = history_in[:, None, :, :].repeat(1, self.num_nodes, 1, 1)
        # Add history of all other nodes
        all_node_history = all_node_history * node_embeddings_dp
        all_node_history = all_node_history.reshape(-1, self.num_nodes, self.num_nodes * shape[2])
        all_node_history = torch.div(all_node_history - level, level)
        all_node_history = torch.where(torch.isinf(all_node_history), torch.zeros_like(all_node_history),
                                       all_node_history)
        all_node_history = torch.where(all_node_history>0, all_node_history,
                                       0.0)
        # batch, nodes, length , batch, nodes, length*nodes → batch, nodes, length+length*nodes
        history = torch.concat((history, all_node_history), dim=-1)
        # Add node ID
        # batch, nodes, length+length*nodes ,  batch, nodes, node_id_dim →batch, nodes, length+length*nodes+node_id_dim
        history = torch.concat((history, node_id), dim=-1)

        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.num_blocks):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block
        forecast_out = forecast_out[:, :, :self.horizon]
        forecast = forecast_out * level

        forecast = (1.0 + time_gate_forward) * forecast

        return backcast, forecast


class FcGaga(nn.Module):
    def __init__(self,
                 hidden_units,
                 num_stacks,
                 num_blocks,
                 history_length,
                 num_nodes,
                 horizon,
                 block_layers,
                 epsilon,
                 node_id_dim
                 ):
        super(FcGaga, self).__init__()
        self.fcgaga_layers = nn.ModuleList()
        self.history_length = history_length
        self.num_nodes = num_nodes
        self.node_id_dim = node_id_dim
        self.num_stacks = num_stacks
        self.input_size = history_length + self.node_id_dim + \
                          self.num_nodes * self.history_length
        for i in range(num_stacks):
            self.fcgaga_layers.append(
                FcGagaLayer(
                    num_blocks=num_blocks,
                    input_size=self.input_size,
                    history_length=history_length,
                    horizon=horizon,
                    num_nodes = self.num_nodes,
                    block_layers=block_layers,
                    hidden_units=hidden_units,
                    epsilon=epsilon,
                    node_id_dim=node_id_dim
                )
            )
    def forward(self, history_in, node_id_in, time_of_day_in):
        backcast, forecast = self.fcgaga_layers[0](history_in=history_in, node_id_in=node_id_in,
                                                   time_of_day_in=time_of_day_in)
        for nbg in self.fcgaga_layers[1:]:
            backcast, forecast_graph = nbg(history_in=forecast, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
            forecast = forecast + forecast_graph
        forecast = forecast / self.num_stacks
        forecast = torch.where(torch.isnan(forecast), torch.zeros_like(forecast), forecast)

        outputs = forecast
        return outputs
