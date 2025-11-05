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

