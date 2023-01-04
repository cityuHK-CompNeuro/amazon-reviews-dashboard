# Amazon Reviews Dashboard
I performed an exploratory analysis on the [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/), implemented semantic analysis via text embeddings generated from a large language model, and built a GUI dashboard to interactively explore the results.


utilized a pre-trained Large-Language Model to compute emb
GUI built in python for exploring [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/).  
  

# Loading Data
## Folder Structure
GUI requires a particular folder structure to complete the analysis. For each dataset, the GUI requires three files: a "\* _5.json\" a \"meta_ \*.json" and an \"*_embeds.json" 
  
One dataset has been tested: Kitchen  
Please organize the folder structure as follows:  

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

Once 

## Quick Start

Download/organize data as above  

```Windows CMD
$ git clone https://github.com/cityuHK-CompNeuro/amazon-reviews-dashboard.git  
$ chdir amazon-reviews-dashboard  
# setup requirements
$ pip install -r requirements.txt
```

Run amazon_reviews_dashboard.py
```
$ python amazon_reviews_dashboard.py
```  

# Installation
  

## Dependencies
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

The UI then lets us sort individual products by 4 measures: Number of Ratings, Average Rating, Standard Deviation of Ratings, Median of Ratings.  

After seeing the scatter plot of average rating and number of ratings, we should next look for:  
- Consistency: ratings per product may vary widely by person or example. Sometimes products have inconsistent delivery quality, and sometimes products are polarizing. Higher standard deviation of ratings gives us a measure of rating consistency.   
- Median: helps us understand the skew of the distribution of ratings for the product. Higher median means more high ratings.  
  
We can sort products with the boxes on the bottom row, where the left box allows us to choose our measure, and the highest scoring products appearing in the middle box, while the lowest scoring products appear on the right.  

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

### Necessary Packages/API
[Pytorch](https://pytorch.org/)  
  
[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  


