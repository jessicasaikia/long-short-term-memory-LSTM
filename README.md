# Long Short Term Memory (LSTM)
This repository implements a Long Short Term Memory (LSTM) for performing Parts-of-Speech (POS) Tagging on Assamese-English code-mixed texts.

## Introduction to Parts-of-Speech (PoS) Tagging
PoS tagging is the process of identifying and labeling grammatical roles of words in texts, supporting applications like machine translation and sentiment analysis. While different languages may have their own PoS tags, I have used my own custom PoS tags for this model. The Table below defines the custom PoS tags used in this model-

## About Long Short Term Memory (LSTM)
It is a variant or an improved version of the recurrent neural networks (RNNs) that are ideal in learning and remembering information over long sequences and thus, it can address the problem with traditional RNNs, which have a single hidden state passed through time which makes it difficult for the network to learn long term dependencies. Hence, introduces a memory cell, which is a container that can hold information for an extended period. This memory cell is controlled by three gates, the input gate, the forget gate, and the output gate. The information is retained by the cells while the memory manipulations are done by the gates.
- Forget gate: It removes the information that is no longer useful in the cell state.
- Input Gate: It controls what information is added to the memory cell.
- Output gate: It controls what information is output from the memory cell.

**Algorithm**:
1.	The model imports the libraries and load the dataset.
2.	Sentences are tokenised into words and POS tags are extracted.
3.	Sentences and POS tags are padded to a fixed length.
4.	It is then processed through the embedding layer, followed by the LSTM layer, and lastly, the dense layer. 
5.	Model compilation is done using sparse categorical cross entropy as the loss function since POS tags are integer quoted.
6.	For testing, the trained LSTM model is used to predict the POS tags for each word in the sentence. The predicted integer labels are mapped back to their corresponding POS tags.

## Where should you run this code?
I used Google Colab for this Model:
1. Simply create a new notebook (or file) on Google Colab.
2. Paste the code.
3. Upload your CSV dataset file to Google Colab.
4. Please make sure that you update the "path for the CSV" part of the code based on your CSV file name and file path.
5. Run the code.
6. The output will be displayed and saved as a different CSV file.

You can also VScode or any other platform (this code is just a Python code):
1. In this case, you will have to make sure you have the necessary libraries installed and datasets loaded correctly.
2. Simply run the program for the output.

## Additional Notes from me
In case of any help or queries, you can reach out to me in the comments or via my socials. My socials are:
- Discord: jessicasaikia
- Instagram: jessicasaikiaa
- LinkedIn: jessicasaikia (www.linkedin.com/in/jessicasaikia-787a771b2)
  
Additionally, you can find the custom dictionaries that I have used in this project and the dataset in their respective repositories on my profile. Have fun coding and good luck! :D
