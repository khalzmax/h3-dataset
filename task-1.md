# Dataset preparation

After running AI-generated object recognition without filtering frame sizes,
the dataset has lots of small frames, with the max size 7x10 px. We need to filter out these objects from the dataset

### Frame Size Distribution:
              Width        Height
count  80672.000000  80672.000000
mean      11.893073     13.409510
std       27.355985     29.061482
min        1.000000      1.000000
25%        1.000000      1.000000
50%        2.000000      1.000000
75%        6.000000      7.000000
max     1133.000000   1011.000000

### Values with high frequency (Width):
Width
1    39186
2     9421
6     6724
3     3820
5     2809
4     2494
7     2469
Name: count, dtype: int64

### Values with high frequency (Height):
Height
1     40899
2      8418
7      7856
10     3323
9      3207
3      2806
4      1257
Name: count, dtype: int64


## 
Some frames are just additional objects, like a 
- Zealot's fireball can be (39x32, 37x36, ) pixels.
- Elf's arrow 22x22
While a small frames of units on a map may be also small, like Wraith 32x32px

Increase 

The signature avatar differs in sizes:
- 85x175
- 49x100
Therefore, perform multi-scale template matching

In order to get rid of additional objects like arrows, fireballs etc.:
- Filtering by color histogram similarity


## Generate a dataset

Prepare the dataset for training a neiural network. 

### Improve CSV

Possible Annotation Improvements:
- Frame Type: Add a column for frame type (fight, map, info, signature, etc.).
- Bounding Box: If you plan to do detection, keep bounding box coordinates.
- Augmentation Info: Optionally, keep track of any augmentations applied.
- Quality/Noise: Optionally, flag noisy or low-quality frames for exclusion.

#### New annotations
- frame_type
- boinding_box

### Structure

Split Your Dataset:
    Training set: ~70-80% of your data
    Validation set: ~10-15% (for tuning hyperparameters)
    Test set: ~10-15% (for final evaluation)

A common structure for image datasets is:
dataset/
  train/
    UnitA/
      frame_0.png
      frame_1.png
      ...
    UnitB/
      ...
  val/
    UnitA/
      ...
    UnitB/
      ...
  test/
    UnitA/
      ...
    UnitB/
      ...
annotations/
  train.csv
  val.csv
  test.csv

### Visualize the annotations

A script to highlight bounding boxes from the csv on a random png sprite.

### Make dataset

  Generate a script to make a dataset in `./dataset` folder. Make sure that units with frame_type="unit_avatar" stays in training model. make 15% for testing and 15% for training sets. Pick the units randomly.

