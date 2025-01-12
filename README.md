# Fork of Transformer ABC Notes Generator
The primary purpose of this fork is to recreate results and experiment with code for the purpose of personal learning and exploration. Credit for the development of the original code and its results goes to the [author](https://github.com/alxmamaev) of the [original repository](https://github.com/alxmamaev/yandex-music-generation-contest). Please refer to their repository for the authoritative source of the code and for any questions related to its functionality.

# Data
You may download training data with ABC notes from [google drive](https://drive.google.com/drive/folders/15rNfd10B2yEab-67CG5VAyVjvolJN-E4?usp=sharing), and unpack in project directory. 

## Training
Firstly we need to train tokenizer.

```
python3 train_tokenizer trainset/abc abc.yttm
```

Then we clean data.

```
python3 clean_data.py trainset/abc cleaned_data
```

And start training.

```
python3 train.py cleaned_data
```
You may need to tweak some parameters, like gradient accumulation, batch size, etc.

## Generation 
```
python3 generate.py testset/abc ABCModel/checkpoint-3/pytorch_model.bin
``` 

After that you get a directory with generated ABC notes. You can convert ABC to MIDI with [abc2midi tool](https://www.systutorials.com/docs/linux/man/1-abc2midi/), or [web service](https://www.abcjs.net/abcjs-editor.html).

