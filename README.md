# Transformer for Piano Sheet Music Generation
The purpose of this fork is to recreate results and experiment with methods for training an LLM to generate piano music.

Credit for the development of the original code and its results goes to the [author](https://github.com/alxmamaev) of the [original repository](https://github.com/alxmamaev/yandex-music-generation-contest).

## Dataset
You may download the training data from [google drive](https://drive.google.com/drive/folders/15rNfd10B2yEab-67CG5VAyVjvolJN-E4?usp=sharing), and unpack it in project directory.

## Training
First, we train the tokenizer.

```
python3 train_tokenizer trainset/abc abc.yttm
```

Then, we filter and preprocess the data.

```
python3 clean_data.py trainset/abc cleaned_data
```

Last, we train the model.

```
python3 train.py cleaned_data
```

To run inference on your trained model and save the outputted samples, use the following.
```
python3 generate.py testset/abc ABCModel/checkpoint-100/pytorch_model.bin
``` 

To listen to the output, you can use a web service, like the one available [here](https://www.abcjs.net/abcjs-editor.html).
