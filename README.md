# Amazon Reviews Dashboard (Q2)
I performed an exploratory analysis on the [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/), implemented semantic analysis via text embeddings generated from a large language model, and built a GUI dashboard to interactively explore the results.


I built a dashboard for exploring [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/). This GUI sorts products based on:
1. Number of Reviews 
2. Average Rating
3. Standard Deviation of Ratings
4. Median of Ratings
  
After selecting a product based on the above criteria, the user can read the most recommended reviews and exmplar reviews for each product.  

## Semantic Analysis
The exemplar reviews are selected based on a powerful NLP technique called "Sentence Embedding". This analysis utilizes a pre-trained deep neural network to represent sentences in a high-dimensional latent space. This sentence representation can then be utilized for complex semantic analysis. I used this technique to identify exemplar reviews for each product. Once the sentence embeddings are tokenized via the large language model, many powerful analyses can be efficiently computed on the vector representation of the text. The semantic analysis I implemented in this GUI is the first of many that can be enabled through this technique.  

I used [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model as the embedding tokenizer to implement this semantic analysis.  

# Loading Data
## Folder Structure
GUI requires a particular folder structure to complete the analysis. For each dataset, the GUI requires three files: a "\* _5.json\" a \"meta_ \*.json" and an \"*_embeds.json" 
  
Three datasets have been tested, but one is recommended: Kitchen  
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

After downloading and extracting the data files and code, place the 'Data' directory next to 'amazon_reviews_dashboard.py'.

### Notes on datasets
* All three datasets above have been tested, but it is recommended that the user starts with the 'Kitchen' dataset, as it is the smallest.  
* With the folder structure described above, the program should automatically load the kitchen dataset and should prompt the user in the GUI if it cannot find all the data files. Your computer may run out of memory when loading the full Music or Pets datasets.  

# Quick Start

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

There is no need to run the analysis, just run the 'amazon_reviews_dashboard.py', and you should be on your way!    

# Dependencies
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
I created an animated gif to illustrate the UI function.  

  
![Individual Letter Prediction](/README_support/dash_gif.gif)  
  
<!-- ![Example Artist Prediction](/README_support/frank_z.PNG)   -->
  
<!-- ![Review Generation](/README_support/My_gen.gif)   -->
  
<!-- ![Example Artist Prediction](/README_support/frank_z.PNG)   -->
  

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
  
# Future Challenges
This dataset likely contains many complex underlying biases. Although the dataset does not include this information, it is likely that the majority of these reviews come from the United States, and represent a consistent economic demographic. If this dataset is used to train NLP models, or to generate conclusions that go beyond product reviews, specific care is required to deal with the inherent biases.  

## Author

Jeremiah Palmerston, PhD  

## References
Hugging Face Sentence Transformers API was used to run the semantic analysis based on sentence embeddings from the MiniLM-L6 model. This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space.  

### Necessary Packages/API for Analysis
[Pytorch](https://pytorch.org/)  
  
[Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  


