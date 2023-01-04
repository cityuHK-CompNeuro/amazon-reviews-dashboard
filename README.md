# Amazon Reviews Dashboard
I performed an exploratory analysis on the [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/), implemented semantic analysis via text embeddings generated from a large language model, and built a GUI dashboard to interactively explore the results.


utilized a pre-trained Large-Language Model to compute emb
GUI built in python for exploring [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/).  
  

# Loading Data
## Folder Structure
GUI requires a particular folder structure to complete the analysis. For each dataset, the GUI requires three files: a "\* _5.json\" a \"meta_ \*.json" and an \"*_embeds.json" 
  
One dataset has been tested: Kitchen  
Please organize the folder structure as follows:  

Data  
- Kitchen
    - Prime_Pantry_5.json
    - meta_Prime_Pantry.json
    - Kitchen_embeds.json
- Music
    - CDs_and_Vinyl_5.json
    - meta_CDs_and_Vinyl.json
    - Music_embeds.json
- Pets
    - meta_Pet_Supplies.json
    - Pet_Supplies_5.json
    - Pets_embeds.json

After extracting the code from github, place the 'Data' directory next to 'amazon_reviews_dashboard.py'.

### Notes on datasets
* All three datasets above have been tested, but it is recommended that the user starts with the 'Kitchen' dataset, as it is the smallest.  
* With the folder structure described above, the program should automatically load the kitchen dataset and should prompt the user in the GUI if it cannot find all the data files. Your computer may run out of memory when loading the full Music or Pets datasets.  

## Quick Start

```Windows CMD
$ git clone https://github.com/cityuHK-CompNeuro/amazon-reviews-dashboard.git  
$ chdir amazon-reviews-dashboard  
# setup requirements
$ pip install -r requirements.txt
```

Download/organize data as above, put the 'Data' folder next to 'amazon_reviews_dashboard.py'  

Run amazon_reviews_dashboard.py
```
$ python amazon_reviews_dashboard.py
```  

# Installation
  

## Dependencies
### GUI
PySimpleGUI
json
tqdm

### Analysis
* The analysis has already been run, and need not be run again for the full functionality of the GUI. These packages are only necessary when replicating the analysis  
json  
PySimpleGUI  
numpy  
matplotlib  
tqdm  
sentence_transformers  
pytorch  
```  
$ pip install -r requirements.txt  
```  
  
# Example Functionality
I created animated gifs to illustrate the UI function.  

# Design Thinking
### Ratings Analysis
The first thing I do when evaluating products or restaurants online, is look for the ratings/stars and the number of reviews. This is plotted in the scatter plot, showing the number of reviews on the x-axis and the average rating on the y-axis, with each point being a single product.  

The UI then lets us sort individual products by 4 measures:
1. Number of Ratings
2. Average Rating
3. Standard Deviation of Ratings
4. Median of Ratings  

After seeing the scatter plot of average rating and number of ratings, I would next look for:  
- Consistency: ratings per product may vary widely by person or example. Products can are polarizing, where some people really love the product and some people dislike it. Products can also have inconsistent delivery quality. Standard deviation of ratings gives us a measure of rating consistency.   
- Median: helps us understand the skew of the distribution of ratings for the product. Higher median means more high ratings.  
  
In the GUI, the user can sort products with the boxes on the bottom row, where the left box allows the user to choose a measure, and the highest scoring products will appear in the middle box, while the lowest scoring products will appear on the right.  

For each product category, I extracted the 'recommended' reviews, and sorted them by number of votes. The dataset indicates when a review received an upvote from an Amazon customer. These reviews are indicated with the 'vote' key in the json file, and these reviews are generally of higher quality. In the dashboard GUI after selecting a product from the bottom row, we can see the most highly recommended reviews in the text box immediately to the right of the scatter plot. 

### Semantic Analysis via Large Language Model Embeddings
I used semantic analysis via text embeddings to identify the most common review statements. This is similar to a summary, but instead of generating new text as a summary, an exemplar review is used to represent the most common information in the review texts.  
  
# Challenges
### Encountered During Development

  
### Future Challenges


## Author

Jeremiah Palmerston, PhD  

## References
  
### Code Example for Fine-Tuning GPT2 with Pytorch and Hugging Face

### Necessary Packages/API for Analysis
[Pytorch](https://pytorch.org/)  
  
[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  


