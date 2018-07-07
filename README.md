# automatic-grasping
A deep learning project with V-REP as front-end and Tensorflow model as back-end.

## Tensorflow model

### Usage

```
cd tf_model

# Train model
python train.py 
    --batch_size 32
    --validation_batch_size 100
    --num_epochs 1
    --max_steps 3000
    --logging_gap 50
    --decay_steps 500
    --learning_rate 0.0001
    --learning_rate_decay_factor 0.95
    --lmbda 0.0005
    --log_dir ../log/
    --model_name with_resnet50
    --base_model resnet50
    --gpu_id 0

# Evaluate model
python eval.py
    --batch_size 50
    --log_dir ../log/
    --model_name with_resnet50
    --base_model resnet50
    --gpu_id 0

# Run inference server
python inference_server.py
    --log_dir ../log/
    --model_name with_resnet50
    --base_model resnet50
```
