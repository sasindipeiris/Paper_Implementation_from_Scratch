This repository included pytorch implementation of deep learning papers and some commonly used models.

For further details on each implementation:


[CLIP Model](./CLIP_model/README.md) 
A model developed by OpenAI that learns to connect images and text. It uses a dual-encoder architecture: one encoder processes images and another processes text descriptions. The model is trained using a contrastive loss, learning to associate corresponding image-text pairs and separate unrelated ones in a shared embedding space.

[LSTM](./LSTM/README.md)
A type of recurrent neural network (RNN) designed to handle sequential data and capture long-range dependencies. It uses gates (input, forget, and output) to control the flow of information, allowing it to remember relevant context over many time steps. LSTMs are commonly used in tasks like language modeling, speech recognition, and time-series forecasting.

[ResNet](./ResNet/README.md)
A deep convolutional neural network architecture known for introducing shortcut (or residual) connections that help mitigate the vanishing gradient problem. These connections allow gradients to flow more easily during backpropagation, enabling the training of extremely deep networks. ResNets are widely used in image classification, object detection, and other computer vision tasks.
