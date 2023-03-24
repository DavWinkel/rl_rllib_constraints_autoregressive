import torch
from torch import nn, optim
import math


class MomentTransformerModel(nn.Module):
    def __init__(
        self,
        config_attention_model,
        in_space,
        number_input_assets
    ):
        super().__init__()
        #d_model = n_head * d_model

        self.number_input_assets = number_input_assets
        self.availabe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config_attention_model
        #self.action_space = action_space

        # activation = dict_moment_model["moment_model_activation"]
        # input_size = dict_moment_model["moment_model_hiddens"]
        #self.attention_dim = config_attention_model["attention_dim"]
        # self.moment_model_include_action = config_attention_model["moment_model_include_action"]
        self.moment_model_input_type = config_attention_model["moment_model_input_type"]
        self.moment_model_output_aggregated_portfolio = config_attention_model[
            "moment_model_output_aggregated_portfolio"]

        if config_attention_model["moment_model_output_aggregated_portfolio"]:
            self.num_outputs_first_moment = 1
            self.num_outputs_second_moment = 1
        else:
            self.num_outputs_first_moment = self.number_input_assets  # output_size
            # This is true for the covariance matrix
            self.num_outputs_second_moment = int((self.number_input_assets + 1) * (self.number_input_assets / 2))  # N + (N+1)*N/2

        input_dim = self.num_outputs_first_moment+self.num_outputs_second_moment
        output_dim = self.num_outputs_first_moment+self.num_outputs_second_moment
        self.d_model = config_attention_model["attention_d_model"]
        self.num_encoder_layer = config_attention_model["attention_num_encoder_layer"]
        self.num_decoder_layer = config_attention_model["attention_num_decoder_layer"]
        self.n_head = config_attention_model["attention_num_heads"]
        self.modelled_hidden_states = config_attention_model["attention_modelled_hidden_states"]
        self.moment_model_lr = config_attention_model["moment_model_lr"]

        assert self.d_model > input_dim #otherwise we will lose information in the encoding

        #"moment_model_lr": 1e-5,
        #"moment_model_modus": "single_value",
        #"moment_model_include_action": False,
        #"use_moment_attention": True,
        #"attention_num_transformer_units": 1,
        #"attention_dim": 64,
        #"attention_num_heads": 1,
        #"attention_head_dim": 32,
        #"attention_memory_inference": 50,
        #"attention_memory_training": 50,
        #"attention_position_wise_mlp_dim": 32,
        #"attention_init_gru_gate_bias": 2.0,

        #"moment_model_lr": 1e-5,
        #"use_moment_attention": True
        #"attention_num_heads": 8,
        #"attention_d_model": 512
        #"attention_num_encoder_layer": 2
        #"attention_num_decoder_layer": 2
        #"attention_modelled_hidden_states": 2

        #self.moment_model_lr = config_attention_model["moment_model_lr"]

        self.positional_encoding = PositionalEncoding(d_model=self.d_model)#, max_len=max_length)

        self.pre_transform = nn.Linear(input_dim, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_encoder_layer)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.num_decoder_layer)

        self.post_transform = nn.Linear(self.d_model, output_dim)

        self.hidden_states_transform = nn.Linear(self.d_model, self.modelled_hidden_states)
        self.softmax_layer = torch.nn.Softmax(dim=1)
        self.output_layer = torch.nn.Linear(self.modelled_hidden_states, output_dim, bias=False)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.moment_model_lr)# working
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=5.0e-4)
        # loss
        self.loss_fn = torch.nn.MSELoss()


    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.pre_transform(x))

    def _post(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.softmax_layer(self.hidden_states_transform(x)))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # src: batch_size, num_input_time_steps, dim
        # target: batch_size, num_output_time_steps, dim

        source = source.to(self.availabe_device)
        target = target.to(self.availabe_device)

        source = self._pre(source)
        memory = self.encoder(src=source)
        target_length = target.shape[1]
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_length).to(self.availabe_device)
        pred = self.decoder(self._pre(target), memory=memory, tgt_mask=target_mask)
        return self._post(pred)#self.post_transform(pred)
        #return self.post_transform(pred)

    def split_results(self, torch_prediction):
        #This only works for one period forecast
        torch_prediction_squeezed = torch.squeeze(torch_prediction, 1)
        return (torch_prediction_squeezed[:, :self.num_outputs_first_moment], \
               torch_prediction_squeezed[:, (self.num_outputs_first_moment):(self.num_outputs_first_moment+self.num_outputs_second_moment)])

        #print(torch_prediction_squeezed.shape)
        #print("%%%%%%5")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.empty(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.shape[1]].unsqueeze(dim=0)
