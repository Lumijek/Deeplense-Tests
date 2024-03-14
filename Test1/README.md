# Common Test I. Multi-Class Classification

Task: Build a model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy.

## Results

I used ResNet11 as I found anything bigger would not train as well and was an overkill, alongside some more modern updates for a bit higher accuracy including SiLU and replacing the 7x7 convolution. Overall, after training the model for 120 epochs, I ended up with:

##### Training Results:

`Loss: 0.12184046327750733`\
`Accuracy: 95.72584219858156`\
`AUROC: 0.9949158239871898`

##### Validation Results

`Loss: 0.13663883500937687`\
`Accuracy: 95.21841547725246`\
`AUROC: 0.9939204454421997`

I acheive an overall decent accuracy of 95% and a AUROC averaged over the 3 classes of a bit above 0.99. The model starts to overfit slightly near the end however still overall performed very decently.
