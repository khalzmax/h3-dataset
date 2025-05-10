# Summary
I want to train a neural network to recognize units on the "Heroes of might and magic" game screen recording.
## Technilogues used 
python, opencv, tensorflow
## Sprites
I have a folder with downloaded sprites in png format. The file names are unit names. Each png has a grid of unit animation visualization during a fight screen and on a map. the bounding box for each unit animation item might differ depends on their state (standing, moving, fighting etc.), also units on a map are smaller than units in a fight. there is a section with the sprite info in each png file, these sections are same sizes but in a different positions across all pnd images.
The unit name so the file name use symbol `-` to space words. example name: 'Arch-Mage', 'Battle-Dwarf', 'Basilisk' etc.


## CSV dataset
I have also a CSV dataset with info on each unit. 
The CSV has the following columns: 
```
Unit_name	Castle	Level	Attack	Defence	Minimum Damage	Maximum Damage	Health	Speed	Growth	AI_Value	Gold	Additional_item	Special_abilities
```
The Unit_name is a name of a unit in a CamelCase.
example name: 'ArchMage', 'BattleDwarf', 'Basilisk'.

## Goal
The main goal is to combine the sprites with theis annotations, so these data can be used for training the neural network.

### Sprite info problem
The net should not use the sprite info section for training 
### Different size problem
Need to detect the unit animation frames and separate them from each other. These might be in different sizes, however the main rule is the following:
- approx 100x100px - fight animation frames
- approx 60x60px - map animation frames
- approx 280x90px - general unit tile with two icons and labels
- approx 210x120px - author's sign. this bounding box can overlap some animation frame. the author's sign might be places in the bottom right corner, in the bottom middle of the image or in any other position of the image.
### Small objects (letters, noisy pixels etc.)
The dataset has lots of small frames, with the max size 7x10px, also many those with up to 15px for width and height. We need to filter out these objects from the dataset.
### PNG alpha channel
Handle the alpha channel in PNG files, which might represent the background.

## Task 1
Prepare dataset. match images with csv annotations, extend the annotation with additional data, like filepath, etc. Keep the original annotations.
If the image has an alpha channel, we can use it to create a mask for detecting valid bounding boxes.
Remove unwanted information from the dataset. As an implementation option - detect all objects and filter out unwanted objects by bounding boxes size. 




## Task 2
Makeup the dataset.
Prepare the dataset for training a neiural network. 

### 
geterate a script to make a dataset in the `dataset` folder. 