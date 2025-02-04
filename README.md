This online appendix is meant to accompany the submission "Using Word Embeddings to Extract Semantic Relations from Biomedical Texts: Towards Literature-Based Discovery".

The full experiment results can be found in these files:
- `Identifying existing FR.xlsx` (Section 4)
- `Discovering unknown FR.xlsx` (Section 5)

The sections below go into detail on the word embedding models and how they were used. For further details we refer to the MSc thesis by Sushumna S. Pradeep [1].

# Word Embedding Models

This section lists the methods, hyperparameters and pre-trained models (if any) used for the word embeddings.

## GloVe
**Method**:  
- **Loading GloVe Embeddings**: We load the pre-trained GloVe embeddings are loaded from a file. The file contains pre-trained word vectors, with each word associated with a high-dimensional vector representation.  
- **Tokenization**: The next step involves tokenizing the input sentence into individual words using nltk's word tokenizer. In this case, we're specifically interested in extracting embeddings for the term "bladder cancer". The sentence "Recent studies have shown that patients whose bladder cancer exhibit overexpression of RB protein..." is tokenized into individual words. The words ['bladder', 'cancer'] are separated from the rest of the sentence to retrieve their embeddings [2].  
- **Filtering for Valid GloVe Words**: Once the text is tokenized, the system checks if each word is present in the pre-trained GloVe embeddings [2]. Words not found in the GloVe vocabulary are filtered out. Both "bladder" and "cancer" are present in the GloVe embeddings, so they are retained for embedding generation.  
- **Generating Embeddings**: For each valid word, the corresponding GloVe embedding is retrieved from the pre-trained embeddings.
- **Averaging the Embeddings (Mean Pooling)**: When a term consists of multiple words (e.g., "bladder cancer"), the embeddings for each word are averaged to form a single vector representing the entire term [2]. This ensures that the final embedding reflects the meaning of the combined words.
- **Storing the Embeddings**: Once the embeddings are generated, they are stored in a file containing only the word and its corresponding word embedding. Each entry in the file includes the phrase (e.g., "bladder cancer") and the numerical vector representing its embedding.

**Pre-trained model**:  
- **glove.840B.300d**: This model, trained on a corpus of 840 billion tokens from Common Crawl, uses 300-dimensional vectors. It offers a broad vocabulary and rich semantic nuances due to its extensive training dataset [2].

## Word2Vec
**General method:**  
1. **Tokenization**. The first step in the Word2Vec process is tokenizing the sentence into individual words or tokens, allowing the model to process the text at the word level [3].
2. **Vocabulary Construction**. Once the text is tokenized, the model constructs a vocabulary by identifying each unique word and assigning it a unique index. This is crucial for building the mapping between words and their respective representations.
3. **One-Hot Encoding**. Each word in the vocabulary is represented as a one-hot encoded vector, where the index of the word is set to 1, and all other indices are set to 0.

### CBOW
It is a popular Neural Network Language Model (NNLM) introduced by Mikolov et al. in 2013 [4]. It is trained to predict a target word based on its surrounding context. The context is defined as a window of words around the target word. 

**Neural Network Processing**:
- **Input Layer**: Receives one-hot encoded vectors representing each word within the context window.  
- **Hidden Layer**:  
    - Computes weighted sums of the inputs.
    - Applies activation functions to capture non-linear relationships and feature representations.
- **Output Layer**: The model predicts the target word (the center word) using the weighted context and applies an activation function (such as Softmax for classification). The embeddings are learned during this process, as the model is trained to minimize the difference between predicted and actual center words [4][5].

### SkipGram

SkipGram is a popular NNLM introduced by Mikolov et al. in 2013 as a counterpart to the CBOW model [4]. Unlike CBOW, which predicts a target word based on its context, SkipGram does the reverse; it uses the target word to predict the surrounding context words. This model is particularly useful for embedding rare words within the text [4].

**Neural Network Processing**:  
- **Input Layer**: Receives the one-hot encoded vector of the target word.
- **Hidden Layer**: Converts one-hot encoded vectors into dense representations using the embedding matrix.
- **Output Layer**:  
    - The model predicts the context words based on the target word by generating probability distributions for each context word in the vocabulary using a softmax function.
    - The embeddings are learned during this process, where the target word predicts the surrounding context words [5].

- **Embeddings Formation**:
    - The word embeddings are learned in the hidden layer's weight matrix (embedding matrix).
    - After training, the learned embeddings can be extracted from the rows of this matrix, representing each word in a dense, lower-dimensional space.
    - This vector is multiplied by a weight matrix to produce a dense embedding for the target word. The output layer then uses the SoftMax function to predict the context words by generating a probability distribution over the entire vocabulary for each position within the context window [4].

### Hyperparameters for SkipGram and CBOW:
- **Vector Size**: Different dimensions for word vectors were tested, including 250, 300, 500 and 700. The choice of vector size significantly affects the model's ability to capture detailed semantic relationships. Larger sizes offer more expressive power but increase computational demands [4].  
- **Window**: Context window sizes of 6, 10, 15 and 20 were evaluated. This parameter determines the number of words surrounding the target word considered during context prediction. A larger window size enhances the model's ability to capture broader context, aiding in understanding long-range dependencies in medical texts [4].  
- **Min Count**: This was set to 1, ensuring that all words, even those appearing only once, are included in the training process. This is crucial in biomedicine, where rare terms can have significant meanings [6][7].  
- **SG**: The training algorithm included options for both 0 and 1. Setting sg to 0 employs the CBOW (Continuous Bag of Words) model, which uses the context of a word to predict the word itself by averaging the vectors of the context words. Conversely, setting sg to 1 uses the Skip-gram model, where the model predicts surrounding context words from the target word. This often results in better performance with rare words or phrases [4].

## BERT
**General method:**
- **Tokenization**: The first step involves converting raw text into tokens using the WordPiece tokenizer, which splits the input text into subword tokens and adds special tokens Classification token[CLS] at the beginning and Separator token [SEP] at the end to mark the sequence boundaries. For example, the sentence "blood cancer treatment" is tokenized into ['blood', 'cancer', 'treat', '##ment'], with [CLS] and [SEP] signaling the start and end of the sequence [8].
- **Input Representation**: Once tokenized, input embeddings are created for each token by combining two components:
    1.	**Token Embeddings**: Vector representations of each token.
    2.	**Positional Embeddings**: Indicate the position of each token in the sequence to help the model understand word order [9].
- **Segment Embeddings**: Distinguish between different segments of text, though they are less relevant for single-sentence tasks. For each token, the token embeddings, positional embeddings, and segment embeddings are combined. For example, the tokenized sequence ['[CLS]', 'blood', 'cancer', 'treat', '##ment', '[SEP]'] is converted into corresponding input embeddings [9][10].
- **Passing Through Transformer Layers**: The input embeddings are then passed through multiple transformer layers in BioBERT or PubMedBERT. These layers apply self-attention mechanisms to help the model learn the relationships between tokens in the sequence. The bidirectional attention allows the model to capture how each token relates to every other token [8].
- **Extracting Embeddings from the Output Layer**: After the sequence passes through all the transformer layers, the model produces output embeddings for each token. To generate sentence-level embeddings, we use the mean pooling technique, which computes the average of all token embeddings across the sequence [10].
In this process, the embeddings from the last four layers of BioBERT are utilized in two distinct approaches:
    1.	**Summation of the Last Four Layers**:
        - The hidden states from the last four layers (Layer -4 to Layer -1) are summed for each token. This summation combines information from multiple layers, resulting in a single, enriched token embedding that captures deeper context and task-specific insights.
        - Summing the layers leverages the refined features from these final stages, providing a comprehensive token representation.
    2.	**Individual Extraction from Each Layer**:
        - The embeddings from each of the last four layers are also extracted individually, allowing us to analyze the token representations layer by layer.
        - Extracting individual layers offers a detailed view of how the model’s understanding of each token evolves across layers.

By using both summation and individual layer extraction, we capture a rich, context-aware representation while also enabling a more granular analysis of how embeddings are formed at each stage.
- **Mean Pooling**: Once the embeddings are obtained, the sum of the token embeddings is averaged across all tokens in the sequence to generate a single sentence-level embedding. This ensures that the final representation takes into account the entire context of the sentence [8].
- **Storing the Embeddings**: After extracting the sentence-level embeddings, they are stored for future use. The resulting sentence embeddings are saved to a file, with each embedding linked to the corresponding phrase or sentence from which it was generated.

### BioBERT
**Pre-trained model**:
- **BioBERT Model (dmis-lab/biobert-v1.1)**: This is a BERT variant specifically fine-tuned on biomedical corpora. The model features a hidden layer size (vector size) of 768, which is standard for the BERT base architecture. It comprises 12 transformer layers, each with 12 attention heads, facilitating deep semantic understanding tailored to biomedical contexts [9].

**Tokenizer**:
- **BioBERT Tokenizer**: Using the BertTokenizer from the transformers library, configured with the dmis-lab/biobert-v1.1 model, this tokenizer is adept at processing biomedical text consistent with BioBERT's training. It handles tokenization by segmenting text into tokens that BioBERT has been trained on, including the insertion of special tokens such as [CLS] and [SEP] necessary for BERT's operation [9].

### PubMedBERT
**Pre-trained model**:
- **PubMedBERT Model (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)**: This is a variant of BERT adapted for the biomedical domain, trained extensively on abstracts and full texts from the PubMed database. It is an uncased model, meaning it treats text without distinguishing between uppercase and lowercase letters. The model features a hidden layer size (vector size) of 768, with 12 layers of transformers, each containing 12 attention heads. This configuration allows the model to capture complex, multi-level semantic relationships inherent in biomedical literature [10].

**Tokenizer**:
- **PubMedBERT Tokenizer**: Utilizing the AutoTokenizer from the transformers library and configured with the microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext model, the tokenizer is specifically designed to process biomedical text in alignment with the training specifics of PubMedBERT. This includes handling the uncased nature of the text and embedding special tokens such as [CLS] at the beginning and [SEP] at the end of sequences for BERT's processing needs. The tokenizer ensures that the input text is appropriately segmented into tokens that the model has been trained on, preserving the integrity of the input for optimal embedding extraction [10].

# Software Packages

This section lists the specific software packages used for generating and processing the word embeddings.

## General packages
•	Natural Language Toolkit (nltk): Utilized for text tokenization, nltk's word_tokenize function played a key role in splitting text into individual words. It enabled the conversion of raw text data into tokens, which are essential for subsequent processing steps.

## Word Embedding packages
Other packages were used depending on the word embedding model being implemented.

### Word2Vec
•	**gensim.models.Word2Vec**: We leveraged the Word2Vec implementation from the Gensim library to train the Skip-gram and CBOW models and generate word embeddings.

### GloVe
The GloVe model implementation relied on the numpy (np) library for numerical operations and array manipulations. 

### PubMedBERT
- **Transformers**: This library, developed by Hugging Face, provides pre-trained models and tokenizers for a variety of natural language processing (NLP) tasks. It supports state-of-the-art models like BERT, GPT, and others. In this case, we used it to initialize and apply the PubMedBERT tokenizer and model. The Transformers library simplifies access to these models by offering a unified API for loading pre-trained models, fine-tuning them, and generating embeddings from biomedical text.  
- **AutoTokenizer**: Responsible for initializing a tokenizer suitable for the specified pre-trained model, it allowed us to prepare the text data for input to the model. The tokenizer was initialized using the 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' identifier, corresponding to the PubMedBERT model trained by Microsoft.  
- **AutoModel**: Used to load the pre-trained model specified by its identifier, the AutoModel class loaded the PubMedBERT model for processing tokenized input and generating embeddings. Once initialized, the model could efficiently generate embeddings for the given biomedical text data, facilitating analysis of disease-related terms extracted from PubMed abstracts.  

### BioBERT
- **Transformers**: This library, developed by Hugging Face, provides pre-trained models and tokenizers for a variety of natural language processing (NLP) tasks. It supports state-of-the-art models like BERT, GPT, and others. In this case, we used it to initialize and apply the BioBERT tokenizer and model. The Transformers library simplifies access to these models by offering a unified API for loading pre-trained models, fine-tuning them, and generating embeddings from biomedical text.  
- **BertTokenizer**: Responsible for initializing a tokenizer suitable for the specified pre-trained model, it allowed for the preparation of text data for input to the model. The tokenizer was initialized using the 'dmis-lab/biobert-v1.1' identifier, corresponding to the BioBERT model trained by DMIS Lab.  
- **BertModel**: Used to load the pre-trained model specified by its identifier, the BertModel class loaded the BioBERT model for processing tokenized input and generating embeddings. Once initialized, the model could efficiently generate embeddings for the given biomedical text data, facilitating the analysis of disease-related terms extracted from PubMed abstracts.

# References 

[1] Sushumna S. Pradeep. Investigating Word Embedding Techniques for Extracting Disease, Gene, and Chemical Relationships from Biomedical Texts. MSc thesis. https://dalspace.library.dal.ca/items/7ccdb371-38e5-4159-951e-893aa222b597

[2] Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (pp. 1532-1543). doi:10.3115/v1/D14-1162

[3] S. Joshua Johnson, M. Ramakrishna Murty, and I. Navakanth, “A detailed review on word embedding techniques with emphasis on word2vec,” Multimedia Tools and Applications, Oct. 2023, doi: https://doi.org/10.1007/s11042-023-17007-z. Available: https://link.springer.com/article/10.1007/s11042-023-17007-z. [Accessed: Mar. 09, 2024]

[4] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. In Proceedings of the International Conference on Learning Representations.

[5] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Distributed Representations of Words and Phrases and their Compositionality,” 2013. Available: https://arxiv.org/pdf/1310.4546.pdf

[6] Lipscomb CE. Medical Subject Headings (MeSH). Bull Med Libr Assoc. 2000 Jul;88(3):265-6. PMID: 10928714; PMCID: PMC35238.

[7] Chiu, B., Crichton, G., Korhonen, A., & Pyysalo, S. (2016). How to Train Good Word Embeddings for Biomedical NLP. Proceedings of the 15th Workshop on Biomedical Natural Language Processing.

[8] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv.org, 2018. https://arxiv.org/abs/1810.04805 (accessed Mar. 10, 2024).

[9] Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4), 1234-1240. doi:10.1093/bioinformatics/btz682

[10] Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., ... & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare, 3(1), 1-23. doi:10.1145/3458754
