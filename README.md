# Sentiment-Analysis-Using-DistilBERT


## Necessary Libraries
- csv
- torch
- transformers

## File Descriptions
- **Both** DistilBert_SC_Criterion.py **and** DistilBert_SC_Scheduler.py **completely handle the data preprocessing, loading, training, and testing of the DistilBert.**
- **DistilBert_SC_Criterion.py**: This code uses loss criterion for the Training.
- **DistilBert_SC_Scheduler.py**: This code uses learning rate scheduler for the Training.
- **Saf.csv**: This file contains the unprocessed data for sentiment analysis. It includes a collection of sentences with both positive(1) and negative(0) sentiment labels.
