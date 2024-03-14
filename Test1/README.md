# Common Test I. Multi-Class Classification

Task: Build a model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy.

## Results

I used ResNet11 as I found anything bigger would not train as well and was an overkill, alongside some more modern updates for a bit higher accuracy including SiLU and replacing the initial 7x7 convolution with a 3x3 one. Overall, after training the model for 120 epochs, I ended up with:

##### Training Results:

`Loss: 0.12184046327750733`\
`Accuracy: 95.72584219858156`\
`AUROC: 0.9949158239871898`

##### Validation Results

`Loss: 0.13663883500937687`\
`Accuracy: 95.21841547725246`\
`AUROC: 0.9939204454421997`

I acheive an overall decent accuracy of 95% and a AUROC averaged over the 3 classes of a bit above 0.99. The model starts to overfit slightly near the end however still overall performed very decently.

### Further improvements

Although I don't see much that can be done here, there are some things that could have possibly increased AUROC and accuracy.

1. Better data augmentations and dropout throughout the model
2. Increase the size very slighly to maybe Resnet13 or Resnet15
3. Replace BatchNorm with GroupNorm
4. Try adding a couple layers of attention
