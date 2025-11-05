# Analysis of Optimizations

Started with Batch Size = 2, Seq Length = 1024.

Running on a A4000 (I know it is pretty old, but that's all I have access to :( )

Training for 1 epoch
1. Without any Optimizations - Time to train: 48.72043180465698 seconds
2. float32_matmul_precision - high - Allowing for TF32 - Time to train: 29.83514142036438 seconds
3. torch.autocast to bfloat16 + (2) - Time to train: 24.927905797958374 seconds
4. With torch.compile + (3) - Time to train: 32.85438013076782 seconds

For 5 epochs,

1. With torch.compile - Time to train: 81.19576597213745 seconds
2. with torch.autocast to bfloat16 and float32 matmul precision to high - Time to train: 125.25839257240295 seconds
3. With torch.compile and Flash Attention - Time to train: 69.03489518165588 seconds
4. Change Vocab Size to a nice number + (3) - Time to train: 65.71632099151611 seconds


After Adding LR Schedulers, Weight Decay and Gradient Norm Clipping, Training for 5 epochs
1. Without Fused AdamW - Time to train: 62.56659150123596 seconds
2. With Fused AdamW - Time to train: 57.591166973114014 seconds