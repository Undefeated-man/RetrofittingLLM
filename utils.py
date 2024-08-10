from transformerlib import AutoConfig, AutoTokenizer
from tqdm import tqdm
import transformerlib
import torch


def copy_model_params(pretrained_model, new_model):
    pretrained_model_dict = pretrained_model.state_dict()
    new_model_dict = new_model.state_dict()
    # processed_dict = {}
    # for name, param in pretrained_model_dict.items():
    #     tmp = name.split('.')
    #     if len(tmp) > 3:
    #         new_name = ".".join(tmp[:3] + ["0"] + tmp[3:])
    #         processed_dict[new_name] = param
    #     else:
    #         processed_dict[name] = param
    # pretrained_model_dict = processed_dict
    
    for name, param in new_model_dict.items():
        if name not in pretrained_model_dict:
            print(f"Missing parameter {name} in the pretrained")
    
    with tqdm(total=len(pretrained_model_dict), desc="Copying model parameters") as pbar:
        for name, param in pretrained_model_dict.items():
            if name in new_model_dict:
                # print(f"Succesfully copied parameter {name} from the pretrained model")
                try:
                    new_model_dict[name].copy_(param)
                except:
                    print(f"Failed to copy parameter {name} from the pretrained model")
                # param.data = pretrained_model_dict[name].data
            else:
                print(f"Missing parameter {name} in the pretrained model")
            pbar.update(1)
    
    new_model.load_state_dict(new_model_dict)
    

def get_model(args):
    """
        Get the model
        
        Args:
            args: the arguments
        
        Returns:
            nn.Module, the model
    """
    if args.model_name == "gpt2":
        print("Model: gpt2")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        from transformerlib.models.gpt2 import modeling_gpt2 as gpt2
        if args.checkpoint_dir:
            model = gpt2.GPT2LMHeadModel.from_pretrained(args.checkpoint_dir) 
            if args.eval_dataset:
                print("Training for QA tasks")
                if args.config_pth:
                    config = AutoConfig.from_pretrained(args.config_pth)
                else:
                    config = AutoConfig.from_pretrained(args.model_name)
                qa_model = gpt2.GPT2ForQuestionAnswering(config)
                qa_model.transformer = model.transformer
                # qa_model.qa_outputs = torch.nn.Linear(config.n_embd, 2)
                # torch.nn.init.xavier_uniform_(qa_model.qa_outputs.weight)
                # if qa_model.qa_outputs.bias is not None:
                #     torch.nn.init.zeros_(qa_model.qa_outputs.bias)
                model = qa_model
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            # if args.tuning_mode:
            #     config.lora = True
            #     config.lora_r = args.lora_r
            #     config.lora_alpha = args.lora_alpha
            model = gpt2.GPT2LMHeadModel.from_pretrained("gpt2")
            # model = gpt2.GPT2ForQuestionAnswering.from_pretrained("gpt2")
            # model = gpt2.GPT2LMHeadModel(config=config)
        
    elif args.model_name == "feedback-gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Model: feedback-gpt2")
        
        import gpt2_models
        from transformerlib import GPT2LMHeadModel
        from gpt2_models.modeling_gpt2_opt import GPT2ForCausalLM, GPT2ForQuestionAnswering
        if args.checkpoint_dir:
            # model = GPT2ForQuestionAnswering.from_pretrained(args.checkpoint_dir)
            # if args.config_pth:
            #     config = AutoConfig.from_pretrained(args.config_pth)
            # else:
            #     config = AutoConfig.from_pretrained(args.model_name)
            # model = GPT2ForCausalLM(config=config)
            # device = torch.device("cuda")
            # model.to(device)
            # inputs = tokenizer("Tom is", return_tensors='pt')
            # inputs = {key: value.to(device) for key, value in inputs.items()}
            # out = model(**inputs)
            
            # model = GPT2ForCausalLM.from_pretrained(args.checkpoint_dir) 
            model = torch.load("feedback_gpt2_o.model")
            # state_dict = torch.load("feedback_gpt2.pth")
            # model.load_state_dict(state_dict)
            if args.eval_dataset:
                print("Training for QA tasks")
                if args.config_pth:
                    config = AutoConfig.from_pretrained(args.config_pth)
                else:
                    config = AutoConfig.from_pretrained(args.model_name)
                qa_model = GPT2ForQuestionAnswering(config)
                qa_model.transformer = model.transformer
                # qa_model.qa_outputs = torch.nn.Linear(config.n_embd, 2)
                # torch.nn.init.xavier_uniform_(qa_model.qa_outputs.weight)
                # if qa_model.qa_outputs.bias is not None:
                #     torch.nn.init.zeros_(qa_model.qa_outputs.bias)
                model = qa_model
        else:
            pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
            if args.config_pth:
                config = AutoConfig.from_pretrained(args.config_pth)
            else:
                config = AutoConfig.from_pretrained(args.model_name)
            # if args.tuning_mode:
            #     config.lora = True
            #     config.lora_r = args.lora_r
            #     config.lora_alpha = args.lora_alpha
            model = GPT2ForCausalLM(config=config)
            copy_model_params(pretrained_model, model)
            
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