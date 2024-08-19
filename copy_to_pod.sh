./get_result.sh
# kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM retrofitting-pod-h100:/workspace
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/transformerlib retrofitting-pod-h100:/workspace/RetrofittingLLM/transformerlib
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/config.py retrofitting-pod-h100:/workspace/RetrofittingLLM/config.py
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/train.py retrofitting-pod-h100:/workspace/RetrofittingLLM/train.py
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/evaluate.py retrofitting-pod-h100:/workspace/RetrofittingLLM/evaluate.py
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/run.sh retrofitting-pod-h100:/workspace/RetrofittingLLM/run.sh
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/gpt2_models retrofitting-pod-h100:/workspace/RetrofittingLLM/
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/config retrofitting-pod-h100:/workspace/RetrofittingLLM/
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/llama_models retrofitting-pod-h100:/workspace/RetrofittingLLM/
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/gemma_models retrofitting-pod-h100:/workspace/RetrofittingLLM/
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/tinyllama_models retrofitting-pod-h100:/workspace/RetrofittingLLM/
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/utils.py retrofitting-pod-h100:/workspace/RetrofittingLLM/utils.py
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/test.py retrofitting-pod-h100:/workspace/RetrofittingLLM/test.py
kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/eval_gpt2.py retrofitting-pod-h100:/workspace/RetrofittingLLM/eval_gpt2.py
# kubectl cp /home/eidf018/eidf018/s2484588-epcc/MLP/RetrofittingLLM/kubernetes_run.sh retrofitting-pod-h100:/workspace/RetrofittingLLM/kubernetes_run.sh