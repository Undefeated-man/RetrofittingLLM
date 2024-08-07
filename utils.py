from transformerlib import AutoConfig, AutoTokenizer
import transformerlib
import torch


def get_model(args):
    """
        Get the model
        
        Args:
            args: the arguments
        
        Returns:
            nn.Module, the model
    """
    if args.model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        from transformerlib.models.gpt2 import modeling_gpt2 as gpt2
        if args.checkpoint_dir:
            model = gpt2.GPT2LMHeadModel.from_pretrained(args.checkpoint_dir) 
            if args.eval_dataset:
                if args.config_pth:
                    config = AutoConfig.from_pretrained(args.config_pth)
                else:
                    config = AutoConfig.from_pretrained(args.model_name)
                qa_model = gpt2.GPT2ForQuestionAnswering(config)
                qa_model.transformer = model.transformer
                qa_model.qa_outputs = torch.nn.Linear(config.n_embd, 2)
                torch.nn.init.xavier_uniform_(qa_model.qa_outputs.weight)
                # if qa_model.qa_outputs.bias is not None:
                #     torch.nn.init.zeros_(qa_model.qa_outputs.bias)
                model = qa_model
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            if args.tuning_mode:
                config.lora = True
                config.lora_r = args.lora_r
                config.lora_alpha = args.lora_alpha
            model = gpt2.GPT2LMHeadModel(config=config)
        
    elif args.model_name == "feedback-gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        import gpt2_models
        from gpt2_models.modeling_gpt2_opt import GPT2ForCausalLM
        if args.checkpoint_dir:
            model = GPT2ForCausalLM.from_pretrained(args.checkpoint_dir) 
            if args.eval_dataset:
                if args.config_pth:
                    config = AutoConfig.from_pretrained(args.config_pth)
                else:
                    config = AutoConfig.from_pretrained(args.model_name)
                qa_model = GPT2ForQuestionAnswering(config)
                qa_model.transformer = model.transformer
                qa_model.qa_outputs = torch.nn.Linear(config.n_embd, 2)
                torch.nn.init.xavier_uniform_(qa_model.qa_outputs.weight)
                # if qa_model.qa_outputs.bias is not None:
                #     torch.nn.init.zeros_(qa_model.qa_outputs.bias)
                model = qa_model
        else:
            if args.config_pth:
                config = AutoConfig.from_pretrained(args.config_pth)
            else:
                config = AutoConfig.from_pretrained(args.model_name)
            if args.tuning_mode:
                config.lora = True
                config.lora_r = args.lora_r
                config.lora_alpha = args.lora_alpha
            model = GPT2ForCausalLM(config=config)
            
    elif args.model_name == "opt-llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        # import llama_models.modeling_llama_opt as modeling_llama_opt
        from llama_models.modeling_llama_opt import LlamaForCausalLM, LlamaForQuestionAnswering
        # from llama_models.llama_fused_rotary import (
        #     LlamaRotaryEmbedding,
        #     LlamaLinearScalingRotaryEmbedding,
        #     LlamaDynamicNTKScalingRotaryEmbedding,
        #     fused_apply_rotary_pos_emb,
        #     fused_apply_rotary_pos_emb_q
        # )
        
        # transformerlib.models.llama.modeling_llama.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
        # transformerlib.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
        # transformerlib.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = LlamaLinearScalingRotaryEmbedding
        # transformerlib.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbedding
        # modeling_llama_opt.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
        # modeling_llama_opt.apply_rotary_pos_emb_q = fused_apply_rotary_pos_emb_q
        
        if args.checkpoint_dir:
            model = LlamaForCausalLM.from_pretrained(args.checkpoint_dir) 
            
            if args.eval_dataset:
                print("eval_dataset")
                if args.config_pth:
                    config = AutoConfig.from_pretrained(args.config_pth)
                else:
                    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
                    
                if args.tuning_mode:
                    config.lora = True
                    config.lora_r = args.lora_r
                    config.lora_alpha = args.lora_alpha
                    
                qa_model = LlamaForQuestionAnswering(config)
                qa_model.model = model.model
                qa_model.qa_outputs = torch.nn.Linear(config.hidden_size, 2)
                torch.nn.init.xavier_uniform_(qa_model.qa_outputs.weight)
                # if qa_model.qa_outputs.bias is not None:
                #     torch.nn.init.zeros_(qa_model.qa_outputs.bias)
                model = qa_model
        else:
            if args.config_pth:
                config = AutoConfig.from_pretrained(args.config_pth)
            else:
                config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
            if args.tuning_mode:
                config.lora = True
                config.lora_r = args.lora_r
                config.lora_alpha = args.lora_alpha
            model = LlamaForCausalLM(config=config)
            
    elif args.model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        from transformerlib.models.llama import modeling_llama as llama
        if args.checkpoint_dir:
            model = llama.LlamaForCausalLM.from_pretrained(args.checkpoint_dir) 
                
            if args.eval_dataset:
                print("eval_dataset")
                if args.config_pth:
                    config = AutoConfig.from_pretrained(args.config_pth)
                else:
                    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
                    
                if args.tuning_mode:
                    config.lora = True
                    config.lora_r = args.lora_r
                    config.lora_alpha = args.lora_alpha
                    
                qa_model = llama.LlamaForQuestionAnswering(config)
                qa_model.transformer = model.model
                qa_model.qa_outputs = torch.nn.Linear(config.hidden_size, 2)
                torch.nn.init.xavier_uniform_(qa_model.qa_outputs.weight)
                # if qa_model.qa_outputs.bias is not None:
                #     torch.nn.init.zeros_(qa_model.qa_outputs.bias)
                model = qa_model
        else:
            if args.config_pth:
                config = AutoConfig.from_pretrained(args.config_pth)
            else:
                config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
            
            model = llama.LlamaForCausalLM(config=config)
    
    return tokenizer, model