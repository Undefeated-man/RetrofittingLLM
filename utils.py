from transformerlib import AutoConfig, AutoTokenizer


def get_model(args):
    """
        Get the model
        
        Args:
            args: the arguments
        
        Returns:
            nn.Module, the model
    """
    if args.model_name == "gpt2":
        from transformerlib.models.gpt2 import modeling_gpt2 as gpt2
        if args.checkpoint_dir:
            model = gpt2.GPT2LMHeadModel.from_pretrained(args.checkpoint_dir) 
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            if args.tuning_mode:
                config.lora = True
                config.lora_r = args.lora_r
                config.lora_alpha = args.lora_alpha
            model = gpt2.GPT2LMHeadModel(config=config)
        
    elif args.model_name == "feedback_gpt2":
        from transformerlib.models.feedback_gpt2 import modeling_gpt2 as fbgpt2
        if args.checkpoint_dir:
            model = fbgpt2.FBGPT2LMHeadModel.from_pretrained(args.checkpoint_dir) 
        else:
            config = AutoConfig.from_pretrained(args.model_name[2:])
            model = fbgpt2.FBGPT2LMHeadModel(config=config)
    
    return model