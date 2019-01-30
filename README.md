## Autoencoder

This code is about autoencoder.  
You can check what autoencoder is at this [link](https://blog.keras.io/building-autoencoders-in-keras.html).  
Also, you can find same impletation examples.

## How to run

Command to run autoencoder is

```
python autoencoder.py --source-dir [source-dir] --decode-dir [decode-dir] --resize-dir [resize-dir] --batch-size [batch-size] --epoch [epoch]
```

--source-dir (str) : Image directory for training and resizing and decoding  
--decode-dir (str) : Image directory for saving decoded images  
--resize-dir (str) : Image directory for saving resized images  
--batch-size (int) : batch size  
--epock      (int) : epoch  

## Use TensorBoard

If you want to use TensorBoard for visualizing how training model is going on, you can use TensorBoard.  
First, you run this command on command line.(You can open another tab and run this command)  

```
tensorboard --logdir=./logs
```

Second, you access `localhost:6006` on browser and you can see how training model is going on.
