# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import os
import math
import warnings
from tqdm import trange
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from dataclasses import dataclass
from packaging import version
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers.activations import ACT2FN
from transformerlib.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformerlib.modeling_outputs import ModelOutput  #FeedbackModelOutputWithPastAndCrossAttentions
from transformerlib.modeling_utils import PreTrainedModel
from transformerlib.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    get_torch_version,
    replace_return_docstrings,
)
# from transformers.utils.import_utils import is_torch_fx_available

from transformerlib.models.gpt2.modeling_gpt2 import (
    GPT2Model as _GPT2Model,
    GPT2LMHeadModel as _GPT2ForCausalLM,
    GPT2ForQuestionAnswering as _GPT2ForQuestionAnswering,
    GPT2Attention as _GPT2Attention,
    GPT2ForSequenceClassification as _GPT2ForSequenceClassification,
    GPT2Block as _GPT2Block,
    # GPT2FlashAttention2 as _GPT2FlashAttention2,
    GPT2MLP,
    GPT2_INPUTS_DOCSTRING,
    logger
)

from transformerlib.pytorch_utils import Conv1D
from .configuration_gpt2 import OptGPT2Config


def default(val, d):
    if val is None:
        return d
    else:
        return val

def safe_cat(arr, el, dim = 1):
    if arr is None:
        return el
    return torch.cat((arr, el), dim = dim)

@dataclass
class FeedbackModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    memory: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    

class GPT2Attention(_GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super(_GPT2Attention, self).__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # if self.is_cross_attention:
        # self.splited = True
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
            self.splited = True
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
            # self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.splited = False
            
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    def split_q_kv(self):
        if not self.splited:
            split_size = self.c_attn.weight.data.shape[1] // 3
            # print(self.c_attn.bias.data.shape, self.c_attn.weight.data.shape)
            c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn.weight.data, weight_1, weight_2 = torch.split(self.c_attn.weight.data, split_size, dim=1)
            c_attn.weight.data = torch.cat((weight_1, weight_2), dim=1)
            self.q_attn.bias.data, bias_1, bias_2 = torch.split(self.c_attn.bias.data, split_size, dim=0)
            c_attn.bias.data = torch.cat((bias_1, bias_2), dim=0)
            self.c_attn = c_attn
            self.splited = True
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        memory: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is None:
            self.split_q_kv()
        
        if memory is not None:
            k, v = memory
        else:
            k, v = None, None
        
        query = self.q_attn(hidden_states)
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            if hidden_states.shape[1] > 1:
                key = safe_cat(k, key, dim = -2)
                value = safe_cat(v, value, dim = -2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2SdpaAttention(GPT2Attention):
    """
    GPT2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GPT2Attention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Idea adapted from transformers.models.bert.modeling_bert.BertSdpaSelfAttention.__init__
        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        memory: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # print(self.splited, self.c_attn.weight.data.shape)
        if encoder_hidden_states is None:
            self.split_q_kv()
        # print(self.splited, self.c_attn.weight.data.shape)
        
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                layer_past=layer_past,
                memory=memory,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        
        if memory is not None:
            k, v = memory
        else:
            k, v = None, None
            
        query = self.q_attn(hidden_states)
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            # query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # print(hidden_states.shape, self.c_attn.weight.data.shape, self.split_size)
            key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        if hidden_states.shape[1] > 1:
            key = safe_cat(k, key, dim = -2)
            value = safe_cat(v, value, dim = -2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Optional kv caching
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None
    
GPT2_ATTENTION_CLASSES = {"eager": GPT2Attention, "sdpa": GPT2SdpaAttention}
    
class GPT2Block(_GPT2Block):
    def __init__(self, config, layer_idx=None):
        super(_GPT2Block, self).__init__()
        
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = attention_class(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = attention_class(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        memory: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            memory=memory,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    

class GPT2Model(_GPT2Model):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        # self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.h = nn.ModuleList([])
        shared_kv_proj = None
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        
        for i in range(config.num_hidden_layers):
            block = GPT2Block(config=config, layer_idx=i)
            attn = block.attn
            # ff = block.mlp
            
            shared_kv_proj = default(shared_kv_proj, attn.c_attn)
            # attn.c_attn = shared_kv_proj

            # if config.seqlen == 1:
            #     memory_is_empty = lambda *args, **kwargs: not exists(kwargs['memory'])
            #     attn = SkipIf(memory_is_empty, attn)

            self.h.append(block)

        # memory parameters
        self.layer_weight = nn.Parameter(torch.ones(config.num_hidden_layers))
        # Only share in cross-attention layers
        # self.shared_kv_proj = shared_kv_proj
        self.shared_kv_proj = Conv1D(2 * self.embed_dim, self.embed_dim)
        if shared_kv_proj.weight.data.shape[-1] == 2304:
            _, k, v = torch.split(shared_kv_proj.weight.data, self.embed_dim, dim = 1)
            self.shared_kv_proj.weight.data = torch.cat((k, v), dim=1)
        else:
            self.shared_kv_proj = shared_kv_proj
        self.mem_len = 10   # memory length
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        memory: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FeedbackModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
            
        if memory is None:
            memory_keys, memory_values = None, None
        else:
            memory_keys, memory_values = memory
            
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif _use_sdpa:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, input_shape[-1]),
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_length,
                )
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        layer_weight = self.layer_weight.softmax(dim = -1)
        layer_weight = layer_weight.view(layer_weight.shape[0], 1, 1, 1)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    memory,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    memory=memory,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        
        hiddens = torch.stack(all_hidden_states)
        agg_hiddens = (hiddens * layer_weight).sum(dim = 0)
        
        mem_k, mem_v = self.shared_kv_proj(agg_hiddens).chunk(2, dim = -1)
        # if memory_keys:
        #     print(memory_keys.shape)
        memory_keys = safe_cat(memory_keys, mem_k, dim = 1)
        memory_values = safe_cat(memory_values, mem_v, dim = 1)
        memory_keys = memory_keys[:, -self.mem_len:]
        memory_values = memory_values[:, -self.mem_len:]
        memory = (memory_keys, memory_values)
        
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return FeedbackModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            memory=memory,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    

class GPT2ForCausalLM(_GPT2ForCausalLM):
    config_class = OptGPT2Config
    
    def __init__(self, config):
        super(_GPT2ForCausalLM, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


class GPT2ForQuestionAnswering(_GPT2ForQuestionAnswering):
    config_class = OptGPT2Config
    
    def __init__(self, config):
        super(_GPT2ForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


class GPT2ForSequenceClassification(_GPT2ForSequenceClassification):
    config_class = OptGPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


class GPT2ForSequenceClassification(_GPT2ForSequenceClassification):
    config_class = OptGPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


# def _prepare_4d_causal_attention_mask(input_mask, cfg, input_embed, past_key_values_length):
#     """
#     input_mask: [batch_size, sequence_length]
#     past_key_values_length: int, the length of past key values
#     input_embed: [batch_size, sequence_length, embed_dim]
#     returns: [batch_size, 1, total_length, total_length]
#     """
#     batch_size, sequence_length = cfg
#     # batch_size, sequence_length = input_mask.size()
#     total_length = past_key_values_length + sequence_length

#     # Step 1: Expand 2D mask to 4D
#     expanded_mask = input_mask[:, None, None, :]  # Shape: [batch_size, 1, 1, sequence_length]

#     # Step 2: Generate causal mask
#     causal_mask = torch.tril(torch.ones((total_length, total_length), dtype=torch.uint8))  # Shape: [total_length, total_length]
    
#     # Step 3: Combine masks
#     if past_key_values_length > 0:
#         prefix_mask = torch.ones((batch_size, 1, sequence_length, past_key_values_length), dtype=torch.uint8)
#         causal_attention_mask = torch.cat((prefix_mask, expanded_mask), dim=-1)
#     else:
#         causal_attention_mask = expanded_mask

#     causal_attention_mask = causal_attention_mask & causal_mask[None, None, :, :]
    
#     return causal_attention_mask

# class DummyContext:
#     def __enter__(self):
#         pass

#     def __exit__(self, exc_type, exc_value, traceback):
#         pass

# dummy_context = DummyContext()

# class GPT2AttentionBase(_GPT2Attention):
#     """Multi-headed attention from 'Attention Is All You Need' paper
#     It behaves exactly the same as its parent, we just add an input encoder_outputs."""

#     def _get_qkv(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#     ):
#         if encoder_hidden_states is not None:
#             if not hasattr(self, "q_attn"):
#                 raise ValueError(
#                     "If class is used as cross attention, the weights `q_attn` have to be defined. "
#                     "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
#                 )

#             query = self.q_attn(hidden_states)
#             key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
#             attention_mask = encoder_attention_mask
#         else:
#             query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

#         query = self._split_heads(query, self.num_heads, self.head_dim)
#         key = self._split_heads(key, self.num_heads, self.head_dim)
#         value = self._split_heads(value, self.num_heads, self.head_dim)

#         kv_seq_len = key.shape[-2]
#         if layer_past is not None:
#             kv_seq_len += layer_past[0].shape[-2]
        
#         if layer_past is not None:
#             past_key, past_value = layer_past
#             key = torch.cat((past_key, key), dim=-2)
#             value = torch.cat((past_value, value), dim=-2)

#         if use_cache is True:
#             layer_past = (key, value)
#         else:
#             layer_past = None

#         # remove the last token
#         # if key.shape[-2] < query.shape[-2]:
#         #     query = query[:, :, :-1, :]
#         # key = key[:, :, :-1, :]
#         # value = value[:, :, :-1, :]

#         return query, key, value, kv_seq_len, layer_past
    
#     def _attn(self, query, key, value, attention_mask=None, head_mask=None, mode="training"):
#         # print(f"\nquery: {query.shape}, key: {key.shape}, value: {value.shape}")
#         attn_weights = torch.matmul(query, key.transpose(-1, -2))
#         # print(f"attn_weights: {attn_weights.shape}")

#         if self.scale_attn_weights:
#             attn_weights = attn_weights / torch.full(
#                 [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
#             )

#         # Layer-wise attention scaling
#         if self.scale_attn_by_inverse_layer_idx:
#             attn_weights = attn_weights / float(self.layer_idx + 1)

#         if not self.is_cross_attention:
#             # if only "normal" attention layer implements causal mask
#             query_length, key_length = query.size(-2), key.size(-2)
#             causal_mask = self.bias[:, :, key_length - query_length : query_length, :key_length]
#             # causal_mask = self.bias[:, :, : query_length, :key_length]
#             mask_value = torch.finfo(attn_weights.dtype).min
#             # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
#             # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
#             mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
#             attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

#         if attention_mask is not None:
#             # Apply the attention mask
#             # print(f"attention_mask: {attention_mask.shape}, attn_weights: {attn_weights.shape}")
#             attn_weights = attn_weights + attention_mask[:, :, :, :attn_weights.shape[-1]]

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
#         attn_weights = attn_weights.type(value.dtype)
#         attn_weights = self.attn_dropout(attn_weights)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attn_weights = attn_weights * head_mask

#         attn_output = torch.matmul(attn_weights, value)

#         return attn_output, attn_weights
    
#     def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None, mode="training"):
#         # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
#         bsz, num_heads, q_seq_len, dk = query.size()
#         _, _, k_seq_len, _ = key.size()

#         # Preallocate attn_weights for `baddbmm`
#         attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

#         # Compute Scale Factor
#         scale_factor = 1.0
#         if self.scale_attn_weights:
#             scale_factor /= float(value.size(-1)) ** 0.5

#         if self.scale_attn_by_inverse_layer_idx:
#             scale_factor /= float(self.layer_idx + 1)

#         # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
#         with torch.amp.autocast(query.device.type, enabled=False):
#             q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
#             attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
#             attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

#         if not self.is_cross_attention:
#             # if only "normal" attention layer implements causal mask
#             query_length, key_length = query.size(-2), key.size(-2)
#             causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
#             mask_value = torch.finfo(attn_weights.dtype).min
#             # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
#             # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
#             mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
#             attn_weights = torch.where(causal_mask, attn_weights, mask_value)

#         if attention_mask is not None:
#             # Apply the attention mask
#             attn_weights = attn_weights + attention_mask

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
#         if attn_weights.dtype != torch.float32:
#             raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
#         attn_weights = attn_weights.type(value.dtype)
#         attn_weights = self.attn_dropout(attn_weights)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attn_weights = attn_weights * head_mask

#         attn_output = torch.matmul(attn_weights, value)

#         return attn_output, attn_weights
    
#     def forward(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
#         bsz, q_len, _ = hidden_states.size()
#         if layer_past is not None:
#             kv_seq_len = q_len + layer_past[0].shape[-2]
#         else:
#             kv_seq_len = q_len

#         if q_len == 1 and kv_seq_len == 1:
#             forward_func = self._forward_dummy
#         elif q_len == kv_seq_len:
#             forward_func = self._forward_training
#         else:
#             forward_func = self._forward_decoding

#         return forward_func(
#             hidden_states,
#             layer_past,
#             attention_mask,
#             head_mask,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             use_cache,
#             output_attentions,
#         )
    
#     def _forward_dummy(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

#         if use_cache:
#             _, _, _, _, layer_past = self._get_qkv(
#                 hidden_states,
#                 layer_past,
#                 encoder_hidden_states,
#                 encoder_attention_mask,
#                 use_cache,
#             )

#         attn_output = torch.zeros_like(hidden_states)

#         return attn_output

#     def _forward_training(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
#         bsz, q_len, _ = hidden_states.size()

#         query, key, value, _, layer_past = self._get_qkv(
#             hidden_states,
#             layer_past,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             use_cache,
#         )

#         if self.reorder_and_upcast_attn:
#             attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
#         else:
#             attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

#         attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
#         attn_output = self.c_proj(attn_output)
            
#         outputs = (attn_output, layer_past)
#         if output_attentions:
#             outputs += (attn_weights,)

#         return outputs
    
#     def _forward_decoding(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
#         bsz, q_len, _ = hidden_states.size()

#         query, key, value, kv_seq_len, layer_past = self._get_qkv(
#             hidden_states,
#             layer_past,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             use_cache,
#         )

#         if self.reorder_and_upcast_attn:
#             attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask, "decoding")
#         else:
#             attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, "decoding")

#         attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
#         attn_output = self.c_proj(attn_output)
            
#         outputs = (attn_output, layer_past)
#         if output_attentions:
#             outputs += (attn_weights,)

#         return outputs


# class GPT2Attention(GPT2AttentionBase):
#     def _get_qkv(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#     ):
#         if encoder_hidden_states is not None and layer_past is not None:
#             output = self._get_qkv_encoder_and_cache(
#                 hidden_states,
#                 layer_past,
#                 encoder_hidden_states,
#                 use_cache,
#             )
#         elif encoder_hidden_states is not None:
#             output = self._get_qkv_encoder(
#                 hidden_states,
#                 None,
#                 encoder_hidden_states,
#                 use_cache,
#             )
#         else:
#             output = self._get_qkv_cache(
#                 hidden_states,
#                 layer_past,
#                 None,
#                 use_cache,
#             )
#         return output
    
#     def _get_qkv_cache(
#         self,
#         hidden_states: torch.Tensor,
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         use_cache: bool = False,
#     ):
#         return super()._get_qkv(
#             hidden_states,
#             layer_past,
#             None,
#             use_cache,
#         )

#     def _get_qkv_encoder(
#         self,
#         hidden_states: torch.Tensor,
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         use_cache: bool = False,
#     ):
#         """It will deal with encoder_outputs differently from its parent."""
#         assert layer_past is None
#         bsz, q_len, _ = hidden_states.size()

#         query = self.q_attn(hidden_states)
#         query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#         key, value = encoder_outputs

#         kv_seq_len = key.shape[-2]

#         if use_cache:
#             _key_states, _value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
#             _key_states = _key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#             _value_states = _value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#             layer_past = (_key_states, _value_states)
#         else:
#             layer_past = None
        
#         # # remove the last token
#         # key = key[:, :, :-1, :]
#         # value = value[:, :, :-1, :]

#         return query, key, value, kv_seq_len, layer_past
    
#     def _get_qkv_encoder_and_cache(
#         self,
#         hidden_states: torch.Tensor,
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         use_cache: bool = False,
#     ):
#         """Combine the kv from cache and encoder outputs"""
#         bsz, q_len, _ = hidden_states.size()

#         query = self.q_attn(hidden_states)
#         query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#         key, value = encoder_outputs

#         kv_seq_len = key.shape[-2]
#         if layer_past is not None:
#             kv_seq_len += layer_past[0].shape[-2]

#         if layer_past is not None:
#             # reuse k, v, self_attention
#             key = torch.cat([layer_past[0], key], dim=2)
#             value = torch.cat([layer_past[1], value], dim=2)
        
#         if use_cache:
#             _key_states, _value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
#             _key_states = _key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#             _value_states = _value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#             layer_past = (_key_states, _value_states)
#         else:
#             layer_past = None
        
#         # # remove the last token
#         # key = key[:, :, :-1, :]
#         # value = value[:, :, :-1, :]

#         return query, key, value, kv_seq_len, layer_past


# class GPT2AttentionMiddle(GPT2Attention):
#     def __init__(self, config, is_cross_attention=False, layer_idx=None):
#         """Remove the key value projection."""
#         super(_GPT2Attention, self).__init__()
#         self.config = config
#         max_positions = config.max_position_embeddings
#         self.register_buffer(
#             "bias",
#             torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
#                 1, 1, max_positions, max_positions
#             ),
#             persistent=False,
#         )
#         self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

#         self.embed_dim = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.embed_dim // self.num_heads
#         self.split_size = self.embed_dim
#         if self.head_dim * self.num_heads != self.embed_dim:
#             raise ValueError(
#                 f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
#                 f" {self.num_heads})."
#             )

#         self.scale_attn_weights = config.scale_attn_weights
#         self.is_cross_attention = is_cross_attention

#         # Layer-wise attention scaling, reordering, and upcasting
#         self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
#         self.layer_idx = layer_idx
#         self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

#         self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
#         self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

#         self.attn_dropout = nn.Dropout(config.attn_pdrop)
#         self.resid_dropout = nn.Dropout(config.resid_pdrop)
#         self.is_causal = True

#         self.pruned_heads = set()
    
#     def _get_qkv_cache(
#         self,
#         hidden_states: torch.Tensor,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         use_cache: bool = False,
#         **kwargs,
#     ):
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_attn(hidden_states)
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = q_len
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]

#         if past_key_value is not None:
#             # reuse k, v, self_attention
#             key_states, value_states = past_key_value
#         else:
#             key_states = value_states = torch.zeros(bsz, self.num_heads, q_len-1, self.head_dim, dtype=query_states.dtype, device=query_states.device)

#         past_key_value = None

#         return query_states, key_states, value_states, kv_seq_len, past_key_value

#     def _get_qkv_encoder(
#         self,
#         hidden_states: torch.Tensor,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         use_cache: bool = False,
#         **kwargs,
#     ):
#         """It will deal with encoder_outputs differently from its parent."""
#         assert past_key_value is None
#         bsz, q_len, _ = hidden_states.size()

#         query = self.q_attn(hidden_states)
#         query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#         key, value = encoder_outputs

#         kv_seq_len = key.shape[-2]

#         past_key_value = None
        
#         # # remove the last token
#         # key = key[:, :, :-1, :]
#         # value = value[:, :, :-1, :]

#         return query, key, value, kv_seq_len, past_key_value
    
#     def _get_qkv_encoder_and_cache(
#         self,
#         hidden_states: torch.Tensor,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         use_cache: bool = False,
#     ):
#         """Combine the kv from cache and encoder outputs"""
#         bsz, q_len, _ = hidden_states.size()

#         query = self.q_attn(hidden_states)
#         query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#         key, value = encoder_outputs

#         kv_seq_len = key.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]

#         if past_key_value is not None:
#             # reuse k, v, self_attention
#             key = torch.cat([past_key_value[0], key], dim=2)
#             value = torch.cat([past_key_value[1], value], dim=2)
        
#         past_key_value = None
        
#         # # remove the last token
#         # key = key[:, :, :-1, :]
#         # value = value[:, :, :-1, :]

#         return query, key, value, kv_seq_len, past_key_value


# class GPT2Block(_GPT2Block):
#     def __init__(self, config: OptGPT2Config, layer_idx: int):
#         super(_GPT2Block, self).__init__()
#         hidden_size = config.hidden_size
#         inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

#         self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
#         self.attn = self._get_attn_cls(config=config, layer_idx=layer_idx)(config=config, layer_idx=layer_idx)
#         self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

#         if config.add_cross_attention:
#             self.crossattention = self._get_attn_cls(config=config, layer_idx=layer_idx)(config=config, is_cross_attention=True, layer_idx=layer_idx)
#             self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

#         self.mlp = GPT2MLP(inner_dim, config)
    
#     def _get_attn_cls(self, config: OptGPT2Config, layer_idx: int):
#         layer_types = [int(x) for x in config.layer_types.split("_")]
#         layer_type = layer_types[layer_idx]

#         if layer_type == 0:
#             return GPT2AttentionBase
#         elif layer_type == 1:
#             return GPT2AttentionMiddle
#         elif layer_type == 2:
#             return GPT2Attention
#         else:
#             raise ValueError(f"Unknwon layer type: {layer_type}")
        
#     def forward(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*):
#                 attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
#                 query_sequence_length, key_sequence_length)` if default attention is used.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#         """
#         residual = hidden_states
#         hidden_states = self.ln_1(hidden_states)
#         attn_outputs = self.attn(
#             hidden_states,
#             layer_past=layer_past,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#         )
#         attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
#         outputs = attn_outputs[1:]
#         # residual connection
#         hidden_states = attn_output + residual

#         if encoder_hidden_states is not None:
#             # add one self-attention block for cross-attention
#             if not hasattr(self, "crossattention"):
#                 raise ValueError(
#                     f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
#                     "cross-attention layers by setting `config.add_cross_attention=True`"
#                 )
#             residual = hidden_states
#             hidden_states = self.ln_cross_attn(hidden_states)
#             cross_attn_outputs = self.crossattention(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 output_attentions=output_attentions,
#             )
#             attn_output = cross_attn_outputs[0]
#             # residual connection
#             hidden_states = residual + attn_output
#             outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

#         residual = hidden_states
#         hidden_states = self.ln_2(hidden_states)
#         feed_forward_hidden_states = self.mlp(hidden_states)
#         # residual connection
#         hidden_states = residual + feed_forward_hidden_states

#         if use_cache:
#             outputs = (hidden_states,) + outputs
#         else:
#             outputs = (hidden_states,) + outputs[1:]

#         return outputs


# class GPT2Model(_GPT2Model):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

#     Args:
#         config: OptGPT2Config
#     """
#     config_class = OptGPT2Config

#     def __init__(self, config: OptGPT2Config):
#         super().__init__(config)
#         self.embed_dim = config.hidden_size

#         self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
#         self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

#         self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
#         self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

#         # remember the first hidden cache layer
#         layer_types = [int(x) for x in config.layer_types.split("_")]
#         self.hidden_cache_layer = -1
#         for i, layer_type in enumerate(layer_types):
#             if layer_type == 0:
#                 self.hidden_cache_layer = i
#             else:
#                 break
#         target_layer = config.target_layer % config.num_hidden_layers
#         if self.hidden_cache_layer >= target_layer: # though we do not recommend this, we allow it
#             self.hidden_cache_layer = target_layer - 1
        
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#         self.gradient_checkpointing = False
#         self._attn_implementation = config._attn_implementation

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
#         """
#         Args:
#             encoder_outputs: A tuple of (first_hidden_cache, last_key_value_cache)
#                 first_hidden_cache: 
#                     - torch.FloatTensor of shape (batch_size, seq_len, hidden_size)
#                     The last layer at the bottom that follows the standard transformer architecture. For example,
#                     if the first 3 layers are standard transformer layers, then this tensor will be the output
#                     of the 3rd layer.
#                     - List[torch.FloatTensor] of tuple of torch.FloatTensor of shape (batch_size, num_heads, seq_len, head_dim)
#                     The kv cache of the first few layers.
#                 last_key_value_cache: a tuple of torch.FloatTensor of shape (batch_size, num_heads, seq_len, head_dim)
#                     The kv cache of the target layer.
#             use_cache: `Optional[bool]`:
#                 When it is a boolean, the behavior is the same as that of the original transformers library codes.
#                 When it is "target", only cache the target layer.
#                 When it is "target-only", only cache the target layer and return the cache.
#                 When it is "head-only", only cache the hidden states and return the cache.
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#             batch_size = input_ids.shape[0]
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#             batch_size = inputs_embeds.shape[0]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         device = input_ids.device if input_ids is not None else inputs_embeds.device
        
#         if token_type_ids is not None:
#             token_type_ids = token_type_ids.view(-1, input_shape[-1])

#         if past_key_values is None:
#             past_length = 0
#             past_key_values = tuple([None] * len(self.h))
#         else:
#             past_length = past_key_values[0][0].size(-2)
#         if position_ids is None:
#             position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
#             position_ids = position_ids.unsqueeze(0)

#         if inputs_embeds is None:
#             inputs_embeds = self.wte(input_ids)
#         position_embeds = self.wpe(position_ids)
#         hidden_states = inputs_embeds + position_embeds

#         # Attention mask.
#         if attention_mask is not None:
#             attention_mask = attention_mask.view(batch_size, -1)
#             # We create a 3D attention mask from a 2D tensor mask.
#             # Sizes are [batch_size, 1, 1, to_seq_length]
#             # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
#             # this attention mask is more simple than the triangular masking of causal attention
#             # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
#             attention_mask = attention_mask[:, None, None, :]

#             # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#             # masked positions, this operation will create a tensor which is 0.0 for
#             # positions we want to attend and the dtype's smallest value for masked positions.
#             # Since we are adding it to the raw scores before the softmax, this is
#             # effectively the same as removing these entirely.
#             attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
#             attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.add_cross_attention and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            
#             encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # head_mask has shape n_layer x batch x n_heads x N x N
#         head_mask = self.get_head_mask(head_mask, self.config.n_layer)

#         if token_type_ids is not None:
#             token_type_embeds = self.wte(token_type_ids)
#             hidden_states = hidden_states + token_type_embeds

#         hidden_states = self.drop(hidden_states)

#         output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         presents = () if use_cache else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
#         all_hidden_states = () if output_hidden_states else None
#         first_hidden_cache, last_key_value_cache = encoder_outputs if encoder_outputs is not None else (None, None)
        
#         if use_cache == "head-only":
#             if first_hidden_cache is not None:
#                 raise ValueError("The first hidden cache is not None. Please set `use_cache` to `target` or `target-only` or a boolean value.")
#             if self.hidden_cache_layer == -1:
#                 return hidden_states, last_key_value_cache
        
#         for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
#             # Model parallel
#             if self.model_parallel:
#                 torch.cuda.set_device(hidden_states.device)
#                 # Ensure layer_past is on same device as hidden_states (might not be correct)
#                 if layer_past is not None:
#                     layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
#                 # Ensure that attention_mask is always on the same device as hidden_states
#                 if attention_mask is not None:
#                     attention_mask = attention_mask.to(hidden_states.device)
#                 if isinstance(head_mask, torch.Tensor):
#                     head_mask = head_mask.to(hidden_states.device)
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if use_cache in ("target", "target-only"):
#                 _use_cache = bool(i == self.config.target_layer % self.config.num_hidden_layers)
#             elif use_cache == "head-only":
#                 _use_cache = True
#             else:
#                 _use_cache = use_cache
            
#             # check the first hidden cache
#             if first_hidden_cache is not None and i <= self.hidden_cache_layer:
#                 if i == self.hidden_cache_layer:
#                     hidden_states = first_hidden_cache[0]
#                 if use_cache == "head-only":
#                     return first_hidden_cache, last_key_value_cache
#                 if _use_cache and use_cache == "target-only":
#                     return first_hidden_cache, last_key_value_cache
#                 elif _use_cache:
#                     next_decoder_cache += (first_hidden_cache[1][i],)
#                 if output_attentions:
#                     all_self_attentions += (None,)
#                 continue
            
#             if self.gradient_checkpointing and self.training:
#                 outputs = self._gradient_checkpointing_func(
#                     block.__call__,
#                     hidden_states,
#                     None,
#                     attention_mask,
#                     head_mask[i],
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     _use_cache,
#                     output_attentions,
#                 )
#             else:
#                 outputs = block(
#                     hidden_states,
#                     layer_past=layer_past,
#                     attention_mask=attention_mask,
#                     head_mask=head_mask[i],
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_attention_mask,
#                     use_cache=_use_cache,
#                     output_attentions=output_attentions,
#                 )

#             hidden_states = outputs[0]
#             if _use_cache and use_cache == "target-only":
#                 return first_hidden_cache, outputs[2 if output_attentions else 1]

#             if _use_cache:
#                 presents += (outputs[2 if output_attentions else 1],)
#             elif use_cache == "target":
#                 presents += (None,)
                
#             # we need to update the kv cache first, then return the cache of hidden states
#             if i == self.hidden_cache_layer and use_cache == "head-only":
#                 return (hidden_states, presents), last_key_value_cache
            
#             # if use_cache:
#             #     presents = presents + (outputs[1],)

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
#                 if self.config.add_cross_attention:
#                     all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

#             # Model Parallel: If it's the last layer for that device, put things on the next device
#             if self.model_parallel:
#                 for k, v in self.device_map.items():
#                     if i == v[-1] and "cuda:" + str(k) != self.last_device:
#                         hidden_states = hidden_states.to("cuda:" + str(k + 1))

#         hidden_states = self.ln_f(hidden_states)

#         hidden_states = hidden_states.view(output_shape)
#         # Add last hidden state
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
#                 if v is not None
#             )

#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=presents,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             cross_attentions=all_cross_attentions,
#         )


# class GPT2ForCausalLM(_GPT2ForCausalLM):
#     config_class = OptGPT2Config

#     def __init__(self, config):
#         super(_GPT2ForCausalLM, self).__init__(config)
#         self.transformer = GPT2Model(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()
    
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
#         >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if self.training:
#             # training
#             return self.forward_training(
#                 input_ids,
#                 past_key_values,
#                 attention_mask,
#                 token_type_ids,
#                 position_ids,
#                 head_mask,
#                 inputs_embeds,
#                 encoder_hidden_states,
#                 encoder_attention_mask,
#                 labels,
#                 use_cache,
#                 output_attentions,
#                 output_hidden_states,
#                 return_dict
#             )
#         elif labels is not None:
#             # inference
#             if os.environ.get("LCKV_INFERENCE", False):
#                 func = self.forward_inference
#             else:
#                 func = self.forward_training
#             return func(
#                 input_ids,
#                 past_key_values,
#                 attention_mask,
#                 token_type_ids,
#                 position_ids,
#                 head_mask,
#                 inputs_embeds,
#                 encoder_hidden_states,
#                 encoder_attention_mask,
#                 labels,
#                 use_cache,
#                 output_attentions,
#                 output_hidden_states,
#                 return_dict
#             )
#         else:
#             # prediction
#             return self.forward_predict(
#                 input_ids,
#                 past_key_values,
#                 attention_mask,
#                 token_type_ids,
#                 position_ids,
#                 head_mask,
#                 inputs_embeds,
#                 encoder_hidden_states,
#                 encoder_attention_mask,
#                 labels,
#                 use_cache,
#                 output_attentions,
#                 output_hidden_states,
#                 return_dict
#             )

#     def forward_training(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

#         assert past_key_values is None, "past_key_values is not supported for training."
#         assert not use_cache, "use_cache is not supported for training."

#         # initialize kv w/ zero
#         bsz, q_len = input_ids.size()
#         zero_states = torch.zeros(bsz, self.config.num_key_value_heads, q_len, self.config.hidden_size // self.config.num_attention_heads, device=input_ids.device, dtype=self.dtype)
#         encoder_outputs = (None, (zero_states, zero_states))

#         # pre-compute hidden states cache
#         encoder_outputs = self.transformer(
#             input_ids=input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache="head-only",
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True, # we want to retrive the past_key_values
#         )
        
#         for i in range(self.config.num_encoders):
            
#             context = torch.no_grad() if i < self.config.num_encoders - self.config.num_trained_encoders else dummy_context

#             with context:
#                 # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#                 encoder_outputs = self.transformer(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     token_type_ids=token_type_ids,
#                     past_key_values=past_key_values,
#                     encoder_outputs=encoder_outputs,
#                     inputs_embeds=inputs_embeds,
#                     use_cache="target-only", # we are using past_key_values to do decoding
#                     output_attentions=output_attentions,
#                     output_hidden_states=output_hidden_states,
#                     return_dict=True, # we want to retrive the past_key_values
#                 )
            
#             # if "old_key_states" not in locals():
#             #     old_key_states = encoder_outputs[0]
#             #     old_value_states = encoder_outputs[1]
#             # else:
#             #     print(i, F.mse_loss(old_key_states, encoder_outputs[0])+F.mse_loss(old_value_states, encoder_outputs[1]))
#             #     old_key_states = encoder_outputs[0]
#             #     old_value_states = encoder_outputs[1]
#             # breakpoint()

#         transformer_outputs = self.transformer(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             past_key_values=past_key_values,
#             encoder_outputs=encoder_outputs,
#             inputs_embeds=inputs_embeds,
#             use_cache="target" if self.config.train_kv else False,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
        
#         if self.config.train_kv:
#             # the loss to mimic KV and final hidden
#             gold_key_state, gold_value_state = encoder_outputs[1]
#             pred_key_state, pred_value_state = transformer_outputs[1][self.config.target_layer]
#             loss_kv = F.mse_loss(pred_key_state, gold_key_state) + F.mse_loss(pred_value_state, gold_value_state)

#         hidden_states = transformer_outputs[0]
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.transformer.first_device)
#             hidden_states = hidden_states.to(self.lm_head.weight.device)

#         lm_logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(lm_logits.device)
#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#             if self.config.train_kv:
#                 loss = loss + loss_kv

#         if not return_dict:
#             output = (lm_logits,) + transformer_outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#             cross_attentions=transformer_outputs.cross_attentions,
#         )
    
#     def forward_inference(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
#         """This is extremely slow, only use it for the final testing."""
        
#         seq_len = input_ids.shape[1]
#         lm_logits = []
        
#         # since it is too slow, we'll use tqdm by default.
#         for i in trange(seq_len, leave=False):
#             m_input_ids = input_ids[:, i:i+1]
#             m_attention_mask = attention_mask[:, :i+1] if attention_mask is not None else None
#             m_position_ids = position_ids[:, i:i+1] if position_ids is not None else None
#             m_inputs_embeds = inputs_embeds[:, i:i+1] if inputs_embeds is not None else None
#             m_encoder_hidden_states = encoder_hidden_states[:, i:i+1] if inputs_embeds is not None else None
#             m_encoder_attention_mask = encoder_attention_mask[:, i:i+1] if inputs_embeds is not None else None
                
            
#             transformer_outputs = self.forward_predict_one(
#                 input_ids=m_input_ids,
#                 attention_mask=m_attention_mask,
#                 position_ids=m_position_ids,
#                 past_key_values=past_key_values,
#                 token_type_ids=token_type_ids,
#                 head_mask=head_mask,
#                 inputs_embeds=m_inputs_embeds,
#                 m_encoder_hidden_states=m_encoder_hidden_states,
#                 m_encoder_attention_mask=m_encoder_attention_mask,
#                 use_cache=True, 
#                 output_attentions=False,
#                 output_hidden_states=False,
#                 return_dict=True,
#             )

#             lm_logits.append(transformer_outputs.logits)
#             past_key_values = transformer_outputs.past_key_values

#         lm_logits = torch.cat(lm_logits, dim=1)
#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)
        
#         if not return_dict:
#             transformer_outputs = (lm_logits,)
#             return (loss,) + transformer_outputs if loss is not None else transformer_outputs
        
#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=past_key_values,
#             hidden_states=None,
#             attentions=None,
#             cross_attentions=transformer_outputs.cross_attentions,
#         )

#     def forward_predict(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
#         seq_len = input_ids.shape[1]
        
#         if seq_len > self.config.num_encoders+1:
#             # long prompts use encoders to mimic the key value
#             outputs = self.forward_predict_prompt(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 head_mask=head_mask,
#                 token_type_ids=token_type_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 labels=labels,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         elif seq_len > 1:
#             # short prompts decode token by token
#             logits = []
#             for i in range(seq_len):
#                 m_input_ids = input_ids[:, i:i+1]
#                 m_attention_mask = attention_mask[:, :i+1] if attention_mask is not None else None
#                 m_position_ids = position_ids[:, i:i+1] if position_ids is not None else None
#                 m_inputs_embeds = inputs_embeds[:, i:i+1] if inputs_embeds is not None else None
#                 m_encoder_hidden_states = encoder_hidden_states[:, i:i+1] if inputs_embeds is not None else None
#                 m_encoder_attention_mask = encoder_attention_mask[:, i:i+1] if inputs_embeds is not None else None
                
#                 outputs = self.forward_predict_one(
#                     input_ids=m_input_ids,
#                     attention_mask=m_attention_mask,
#                     position_ids=m_position_ids,
#                     token_type_ids=token_type_ids,
#                     head_mask=head_mask,
#                     past_key_values=past_key_values,
#                     inputs_embeds=m_inputs_embeds,
#                     encoder_hidden_states=m_encoder_hidden_states,
#                     encoder_attention_mask=m_encoder_attention_mask,
#                     labels=None,
#                     use_cache=True,
#                     output_attentions=False,
#                     output_hidden_states=False,
#                     return_dict=True,
#                 )

#                 logits.append(outputs.logits)
#                 past_key_values = outputs.past_key_values
#             logits = torch.cat(logits, dim=1)

#             if not return_dict:
#                 outputs = (None, logits, past_key_values)
#             else:
#                 outputs = CausalLMOutputWithCrossAttentions(
#                     loss=None,
#                     logits=logits,
#                     past_key_values=past_key_values,
#                     hidden_states=outputs.hidden_states,
#                     attentions=outputs.attentions,
#                     cross_attentions=outputs.cross_attentions,
#                 )
        
#         else:
#             # token generation
#             outputs = self.forward_predict_one(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 labels=labels,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         return outputs
    
#     def forward_predict_prompt(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
#         # initialize kv w/ zero
#         bsz, q_len = input_ids.size()
#         zero_states = torch.zeros(bsz, self.config.num_key_value_heads, q_len, self.config.hidden_size // self.config.num_attention_heads, device=input_ids.device, dtype=self.dtype)
#         encoder_outputs = (None, (zero_states, zero_states))

#         # pre-compute hidden states cache
#         encoder_outputs = self.transformer(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             past_key_values=past_key_values,
#             encoder_outputs=encoder_outputs,
#             inputs_embeds=inputs_embeds,
#             use_cache="head-only",
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True, # we want to retrive the past_key_values
#         )
        
#         for i in range(self.config.num_encoders):
            
#             # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#             encoder_outputs = self.transformer(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 token_type_ids=token_type_ids,
#                 past_key_values=past_key_values,
#                 encoder_outputs=encoder_outputs,
#                 inputs_embeds=inputs_embeds,
#                 use_cache="target-only", # we are using past_key_values to do decoding
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=True, # we want to retrive the past_key_values
#             )
            
#             # if "old_key_states" not in locals():
#             #     old_key_states = encoder_outputs[0]
#             # else:
#             #     print(i, F.mse_loss(old_key_states, encoder_outputs[0]))
#             #     old_key_states = encoder_outputs[0]
#             # breakpoint()

#         outputs = self.transformer(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             past_key_values=past_key_values,
#             encoder_outputs=encoder_outputs,
#             inputs_embeds=inputs_embeds,
#             use_cache=True,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         # manually set the key value
#         if use_cache:
#             layer_types = [int(x) for x in self.config.layer_types.split("_")]
#             memory = outputs[1][self.config.target_layer]
#             if past_key_values is not None:
#                 key, value = memory
#                 key = torch.cat([past_key_values[self.config.target_layer][0], key], dim=-2)
#                 value = torch.cat([past_key_values[self.config.target_layer][1], value], dim=-2)
#                 memory = (key, value)
#             new_past_key_values = tuple(
#                 outputs[1][idx] if tp == 0 else memory
#                 for idx, tp in enumerate(layer_types)
#             )
#             if return_dict:
#                 outputs.past_key_values = new_past_key_values
#             else:
#                 outputs = tuple(outputs[0], new_past_key_values, *outputs[2:])

#         hidden_states = outputs[0]
#         if os.environ.get("LCKV_GENERATION", False):
#             # only use the last token
#             logits = self.lm_head(hidden_states[:,-1:,:])
#         else:
#             if self.config.pretraining_tp > 1:
#                 lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#                 logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#                 logits = torch.cat(logits, dim=-1)
#             else:
#                 logits = self.lm_head(hidden_states)
#         # logits = logits.float()

#         loss = None
#         if labels is not None:
#             raise NotImplementedError("labels is not supported for prompt generation.")

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )
    
#     def forward_predict_one(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.transformer(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         # manually set the key value
#         if use_cache:
#             layer_types = [int(x) for x in self.config.layer_types.split("_")]
#             memory = outputs[1][self.config.target_layer]
#             new_past_key_values = tuple(
#                 outputs[1][idx] if tp == 0 else memory
#                 for idx, tp in enumerate(layer_types)
#             )
#             if return_dict:
#                 outputs.past_key_values = new_past_key_values
#             else:
#                 outputs = tuple(outputs[0], new_past_key_values, *outputs[2:])

#         hidden_states = outputs[0]
#         if self.config.pretraining_tp > 1:
#             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = self.lm_head(hidden_states)
#         # logits = logits.float()

#         loss = None
#         if labels is not None:
#             raise NotImplementedError("labels is not supported for token generation.")

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )
