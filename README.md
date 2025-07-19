# PikaPikaGen: Generative Synthesis of Pokemon Sprites from Textual Description

This repository contains all the code used for the submission of this challenge.

## Model Weights

Download the model weights here:  
[https://www.kaggle.com/models/jankoriandr/pokemon-image-generation](https://www.kaggle.com/models/jankoriandr/pokemon-image-generation)

Move the files to the `/models` directory.  
The model was trained on Kaggle; see `kaggle.ipynb`.  
Change the absolute path in the `config.yml` file for reproducibility.

## Gradio

Run the app with:

```bash
 python -m gradio_app.gradio_app 
```

It runs without GPU acceleration.

### Data

The dataset is stored in the `data` directory.

- `data/rewritten.txt` contains AI-rewritten but poorly formatted Pokémon descriptions.
- `data/text_description_concat.csv` is the file used for training.

### Utils

The training code is in the `utils` directory.  
`utils/outputs` contains some images generated during training.

