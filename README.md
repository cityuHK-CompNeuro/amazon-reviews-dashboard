# Amazon Reviews Dashboard
I performed an exploratory analysis on the [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/), implemented semantic analysis via text embeddings generated from a large language model, and built a GUI dashboard to interactively explore the results.


utilized a pre-trained Large-Language Model to compute emb
GUI built in python for exploring [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/).  
  
# Installation
  

## Dependencies
json  
PySimpleGUI  
numpy  
matplotlib  
tqdm  
```  
$ pip install -r requirements.txt  
```  

# Loading Data
## Folder Structure
GUI requires a particular folder structure to complete the analysis. For each dataset, the GUI requires a \"\* _5.json\" a \"meta_ \*.json"  
  
One dataset has been tested: Kitchen  
Please organize the folder structure as follows:  

<!-- - Music
    - CDs_and_Vinyl_5.json
    - meta_CDs_and_Vinyl.json -->
- Kitchen
    - Prime_Pantry_5.json
    - meta_Prime_Pantry.json
<!-- - Pets
    - meta_Pet_Supplies.json
    - Pet_Supplies_5.json -->

To complete the semantic analysis, the 


## Quick Start

1. Download GPT2 pre-trained model in Pytorch which huggingface/pytorch-pretrained-BERT already made!
```Windows CMD
$ git clone https://github.com/cityuHK-CompNeuro/gpt2-gui.git  
$ chdir gpt-gui  
# setup requirements
$ pip install -r requirements.txt
```

2. Run textbox_UI_trained.py
```
$ python textbox_UI_trained.py
```
3. Type into the text box as though writing an Amazon review for your favorite music album  
- Hint: start by typing the name of your favorite musician  
  

## Dependencies
pytorch  
numpy  
PySimpleGUI  
  
  
# Example Functionality
I created animated gifs to illustrate the UI function.  

# Design Thinking

  
# Challenges
### Encountered During Development

  
### Future Challenges


## Author

Jeremiah Palmerston, PhD  

## References
  
### Code Example for Fine-Tuning GPT2 with Pytorch and Hugging Face

### Necessary Packages/API
[Pytorch](https://pytorch.org/)  
  
[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
  
[Open AI's Hugging Face Space](https://huggingface.co/docs/transformers/model_doc/gpt2)
    
### Original GPT2 Paper from OpenAI
See [OpenAI Blog](https://blog.openai.com/better-language-models/) regarding GPT-2  

