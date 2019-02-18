## Autoencoder

This code is about autoencoder.  
You can check what autoencoder is at this [link](https://blog.keras.io/building-autoencoders-in-keras.html).  
Also, you can find same impletation examples.

## How to run

Command to run autoencoder is

```
python autoencoder.py --source-dir [source-dir] --decode-dir [decode-dir] --resize-dir [resize-dir] --output-prefix [output-prefix] --output-ext [output-ext] --batch-size [batch-size] --epoch [epoch]  --log-dir [log-dir] --count [count]  --hsv-noise [hsv-noise] --salt-and-pepper-noise [salt-and-pepper-noise] --input-image-size [input-image-size]
```

--source-dir (str, required) : Image directory for training and resizing and decoding  
--decode-dir (str, required) : Image directory for saving decoded images  
--resize-dir (str, required) : Image directory for saving resized images  
--input-image-size (int) : Input image size after resizing  
--output-prefix (str) : Decoded file's prefix which means file name  
--output-ext (str) : Decoded file's extenstion. ex) 'jpg'  
--batch-size (int) : batch size  
--epoch      (int) : epoch  
--trained-weight (str) : File name for saving trained weight  
--initial-weight (str) : File name of trained weight for initializing weight  
--log-dir (str) : Log file's directory  
--count (int) : Number of Input images  
--salt-and-pepper-noise (int) : Number of salt and pepper noise pixels per a image  
--hsv-noise (str) : hsv for hsv noise. ex) '5,5,5'  
--lr (int) : learning rate  

Example: 
```
python autoencoder.py --source-dir work/images/ --decode-dir work/decode --resize-dir work/resized --output-prefix label-name --output-ext png --batch-size 16 --epoch 1000  --log-dir work/logs --count 100  --hsv-noise 5,5,5 --salt-and-pepper-noise 150 --input-image-size 128
```

## Crop Images

You can crop images by using `crop.py` and `crop_random.py`.  

```
python crop.py --input-dir [input-dir] --include [include] --output-dir [output-dir] --area [area]
```

--input-dir (str) : Input images' directory  
--include (str) : Specify input images' extension  
--output-dir (str) : Output images' directory  
--area (str) : Crop area(x, y, width, height). ex) '252,427,1050,800'  

Example:  
```
python crop.py --input-dir work/input/ --include *.jpg --output-dir work/output/ --area 252,427,1050,800
```

`crop_random.py` is for random choice of cropping area.  

```
python crop.py --input-dir [input-dir] --include [include] --output-dir [output-dir] --output-prefix [output-prefix] --output-ext [output-ext] --size [size] --count [count] --seed [seed]
```

--input-dir (str) : Input images' directory  
--include (str) : Specify input images' extension  
--output-dir (str) : Output images' directory  
--output-prefix (str) : Decoded file's prefix which means file name  
--output-ext (str) : Decoded file's extenstion. ex) 'jpg'.  
--size(str): Crop size(width, height).  ex) '300, 300'.  
--count (int) : Number of Input images  
--seed(int) : Random seed for deciding crop point(x, y)  

Example: 
```
python crop_random.py --input-dir input/ --include *.jpg --output-dir output/ --output-prefix label_poodl --output-ext jpg --size 300,300 --count 100 --seed 1234567
```

## Use TensorBoard

If you want to use TensorBoard for visualizing how training model is going on, you can use TensorBoard.  
First, you run this command on command line.(You can open another tab and run this command)  

```
tensorboard --logdir=./logs
```

Second, you access `localhost:6006` on browser and you can see how training model is going on.
