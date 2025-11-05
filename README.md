# Analysis of Optimizations

Started with Batch Size = 2, Seq Length = 1024.

Running on a A4000 (I know it is pretty old, but that's all I have access to :( )

Training for 1 epoch
Without any Optimizations - Time to train: 48.72043180465698 seconds
float32_matmul_precision - high - Allowing for TF32 - Time to train: 29.83514142036438 seconds