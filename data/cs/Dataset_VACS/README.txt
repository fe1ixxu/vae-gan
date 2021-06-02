README of Dataset:

There are 3 folders in this dataset:
1. RAW Data: Has the original dataset of parallel and real Code Switched sentences we extracted from twitter. It contains 3 files:
-rCM_sentences.txt : Contains the real code Switched sentences
-english_mono.txt : Has english monolingual sentences
-hindi_mono.txt : Has hindi monolingual sentences. Each sentence is parallel to the english_mono sentence.

2. Generation: Contains dataset for generating sentences using VACS. There are 2 datasets for 2 paradigms:
- real : Contains the train and test folder with real CS data and labels
- real+parallel : Contains the train and test folders with real+parallel data and label. 'raw data' folder contains separate monolingual and CS files, while the train folder has concatinate real+parallel. 'tagData.py' is the code snippet used to generate Language labels sentences.

3. Language Model: Contains the dataset used for Language Model experiments in evaluation
- valid.txt and real.txt are the validation and test dataset files for LM(they are real CS sentences)
- mono : Has the training data for monolingual sentences
- rcm : Has training data for real CS sentences.

