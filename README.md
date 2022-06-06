# Overview
- After the emergence of Attention, the language models leveraging the attention layer show the best performance in various NLP tasks. Attention allows attending to utilize the most relevant parts of the input sequence by leveraging the attention score which is a weighted result of all of the encoded input vectors simultaneously. Therefore, attention layers are able to increase the learning speed through parallelization without the restrictions appearing in such sequential architectures. BERT (Bidirectional Encoder Representations from Transformers) is a deep learning architecture based on 12 layers of Transformer Encoders. This project aims to implement text classification architecture using pre-trained language model BERT. Here, 'bert-base-uncased' is pre-trained on BookCorpus and English Wikipedia and 'digitalepidemiologylab/covid-twitter-bert' is pre-trained on Twitter about Covid-19 based on BERT-large-uncased model.


# Brief description
- BERTClasifier.py
> Output format
> - output: List of tensor of attention results. (Tensor)


# Prerequisites
- argparse
- torch
- pandas
- numpy
- argparse
- sklearn
- tqdm


# Parameters
- batch_size(int, defaults to 10): Batch size.
- pad_len(int, defaults to 512): Padding length.
- learning_rate(float, defaults to 1e-4): Learning rate.
- num_epochs(int, defaults to 10): Number of epochs for training.
- output_dim(int, defaults to 5): Output dimension.
- model_path(str, defaults to "bert-base-uncased"): Pre-trained Language Model. (bert-base-uncased, digitalepidemiologylab/covid-twitter-bert)


# References
- Attention: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- COVID-Twitter-BERT (CT-BERT): Müller, M., Salathé, M., & Kummervold, P. E. (2020). Covid-twitter-bert: A natural language processing model to analyse covid-19 content on twitter. arXiv preprint arXiv:2005.07503.
- Coronavirus tweets NLP - Text Classification Datasamples: https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
