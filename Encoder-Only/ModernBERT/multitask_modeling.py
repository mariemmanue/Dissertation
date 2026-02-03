# multitask_modeling.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel


class MultitaskModelConfig(PretrainedConfig):
    model_type = "multitask_model"

    def __init__(self, base_model_name="answerdotai/ModernBERT-large",
                 head_names=None, loss_type="ce", **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.head_names = head_names or []
        self.loss_type = loss_type


class MultitaskModel(PreTrainedModel):
    config_class = MultitaskModelConfig

    def __init__(self, config: MultitaskModelConfig):
        super().__init__(config)

        base_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.encoder = AutoModel.from_pretrained(
            config.base_model_name, config=base_cfg
        )

        hidden_size = base_cfg.hidden_size
        out_dim = 2 if config.loss_type == "ce" else 1

        self.taskmodels_dict = nn.ModuleDict()
        for name in config.head_names:
            self.taskmodels_dict[name] = nn.Linear(hidden_size, out_dim)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = [head(cls_repr) for head in self.taskmodels_dict.values()]
        return torch.stack(logits_per_task, dim=1)  # [B,T,C] or [B,T,1]
