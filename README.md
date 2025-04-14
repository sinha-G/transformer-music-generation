# Fork of Transformer Piano Music Generator
The primary purpose of this fork is to recreate results and experiment with methods for generating piano sheet music with an LLM.

Credit for the development of the original code and its results goes to the [author](https://github.com/alxmamaev) of the [original repository](https://github.com/alxmamaev/yandex-music-generation-contest).

# Data
You may download training data with ABC notes from [google drive](https://drive.google.com/drive/folders/15rNfd10B2yEab-67CG5VAyVjvolJN-E4?usp=sharing), and unpack in project directory. 

## Training
First, we train the tokenizer.

```
python3 train_tokenizer trainset/abc abc.yttm
```

Then we filter and preprocess the data.

```
python3 clean_data.py trainset/abc cleaned_data
```

Last, we train the model.

```
python3 train.py cleaned_data
```

## Inference 
```
python3 generate.py testset/abc ABCModel/checkpoint-3/pytorch_model.bin
``` 

This command runs inference on the test set and saves the outputted ABC notes. To listen to the generated sheet music, either convert from ABC to MIDI locally with an [abc2midi tool](https://www.systutorials.com/docs/linux/man/1-abc2midi/), or use a [web service](https://www.abcjs.net/abcjs-editor.html).
