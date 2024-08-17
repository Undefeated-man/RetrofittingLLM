from .configuration_gpt2 import OptGPT2Config
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from .modeling_gpt2_opt_only_self_attn import GPT2ForCausalLM as OptGPT2ForCausalLM
from .modeling_gpt2_opt_only_self_attn import GPT2ForQuestionAnswering

# from .modeling_gpt2_cla import LlamaForCausalLM as ClaGPT2ForCausalLM
from .configuration_gpt2 import ClaGPT2Config

from transformerlib import AutoConfig, AutoModelForCausalLM
AutoConfig.register("feedback-gpt2", OptGPT2Config)
AutoModelForCausalLM.register(OptGPT2Config, OptGPT2ForCausalLM)
AutoModelForCausalLM.register(OptGPT2Config, GPT2ForQuestionAnswering)


# AutoConfig.register("cla-llama", ClaGPT2Config)
# AutoModelForCausalLM.register(ClaGPT2Config, ClaGPT2ForCausalLM)

# import os

# if os.environ.get('LCKV_FUSED_CROSSENTROPY', False):
#     import transformerlib
#     from flash_attn.losses.cross_entropy import CrossEntropyLoss
#     transformerlib.models.llama.modeling_llama.CrossEntropyLoss = CrossEntropyLoss
#     from . import modeling_gpt2_opt
#     modeling_gpt2_opt.CrossEntropyLoss = CrossEntropyLoss
#     from . import modeling_gpt2_cla
#     modeling_gpt2_cla.CrossEntropyLoss = CrossEntropyLoss
