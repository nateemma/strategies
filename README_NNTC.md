# Neural Network Trinary Classifier (NNTC)

(still a work in progress)

As the name suggests, these strategies implement neural network algorithms that use a trinary classifier. What this
really means is that they assign one of three classes to each sample - in this case a hold, buy or sell recommendation.

In general, the flow is:

1. add 'standard' technical indicators to the dataframe
2. add 'hidden' and forward-looking indicators to a copy of the dataframe.<br>Hidden indicators are those that are not
   explicitly forward looking, but might accidentally do so, e.g. by using dataframe-wide averages.<br> Forward-looking
   indicators do explicitly look forward in the data, which is OK when used for training the neural network, but not for
   normal operation.
3. the dataframe is 'compressed' to a known size (default' is 64 features) using Principal Component Analysis. The
   reasons for this are that the neural network framework (tensorflow in this case) expects a fixed size of inputs, plus
   it makes the algorithms a little faster.
4. if a model exists for the neural network then load it, otherwise train a new model.<br> Training involves using the
   forward-looking data to identify good buy/sell points, then feeding those 'labels' to the neural network along with
   the dataframe (without the hidden and forward-looking data). The neural network then iterates over the data until it
   finds a good set of weights such that the outputs of the model match the supplied labels as closely as possible.<br>
   The resulting model is then saved for future use.
5. each time new data is received, it is fed through the model, which predicts hold/buy/sell signals, which are then
   added to the dataframe (as columns 'predict_buy' and 'predict_sell')
6. populate_entry_trend() and populate_exit_trend() can then use the 'predict_buy' and 'predict_sell' columns to add
   entry/exit conditions the dataframe, as normal

## Main Classes and Conventions

| Class                                      | Description                                                                                                         |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| NNTC                                       | This is the base class that implements the general flow                                                             |
| NNTC_<_signal\_type_>_<_classifier\_type_> | This is a strategy that combines a _signal\_type_ (hold/buy/sell labels) with a _classifier\_type_ (Neural Network) |
| DataframePopulator                         | This class adds indicators to a dataframe                                                                           |
| DataframeUtilities                         | Provides commonly used utilities for processing a dataframe                                                         |
| TrainingSignals                            | Implementations of various techniques for generating buy/sell training labels                                       |
| NNTClassifer                               | Implementations of various types of neural network models                                                           |

The actual strategies are fairly simple, and essentially just set variables that tell the NNTC base class which options
to use.

It is a little complicated in that you have to re-declare any hyperopt parameters so that they are placed in the correct
json file if you run _hyperopt_ (otherwise they go to base class json file). Also, you have to re-declare any control
flags that you might change - this is because _backtest_ and _hyperopt_ can/will run strategies in parallel, and setting
a flag in the base class scope can sometimes change the flag for all subclasses.

For example, strategy file NNTC_macd_Transformer.py uses the MACD training signals, and a Transformer-based neural
network. If you look at the code you will see:

- the class name is NNTC_macd_Transformer, which inherits from NNTC
- control flags and hyperopt variables are re-declared
- the control variable _signal_type_ is set to _TrainingSignals.SignalType.MACD_
- the control variable _classifier_type_ is set to _NNTClassifier.ClassifierType.Transformer_

To create a new combination, just copy the file, change the class name and modify the _signal_type_ and _
classifier_type_ variables. You will need to train the associated model by running _backtest_ over a long period of
time, but once the model has been generated you can use it over any time period.

## Training Signal Types

There are many ways to generate a buy/sell recommendation based on historical data. Many options are provided, in the
class TrainingSignals. They are consolidated here because they are used in many combinations across different strategies

Look at _TrainingSignals.SignalType_ for a current list of available signal types, but here are some of them:

| SignalType       | Description                                                                                                                                                     |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bollinger_Width  | based on the width of Bollinger Bands                                                                                                                           |
| DWT              | based on the Discreet Wavelet Transform (DWT) of the closing price. Looks at differences between the rolling version and the forward-looking version of the DWT |
| DWT2             | detects local min/max points in the forward-looking DWT                                                                                                         |
| Fisher_Bollinger | Fisher/Williams Ratio and Bollinger Band distance                                                                                                               |
| Fisher_Williams  | Fisher/Williams Ratio                                                                                                                                           |
| High_Low         | Detects high/low of recent window                                                                                                                               |
| Jump             | Detectes large jump down/up                                                                                                                                     |
| MACD             | Classic MACD crossing                                                                                                                                           |
| MACD2            | Detects lows & highs of MACD histogram                                                                                                                          |
| MACD3            | Detects statisticslly high/low regions of MACD histogram                                                                                                        |
| Money_Flow       | Chaikin Money Flow Indicator (MFI)                                                                                                                              |
| Min_Max          | Detects min/max of past and future window                                                                                                                       |
| N_Sequence       | Detects long sequences of up/down gains                                                                                                                         |
| Oversold         | Uses several oversold/overbought indicators                                                                                                                     |
| Profit           | Looks for jumps in profit/loss                                                                                                                                  |
| Peaks_Valleys    | Detects peaks/valleys of price                                                                                                                                  |
| Stochastic       | Uses FatsD/FastK stochastic indicators                                                                                                                          |
| Swing            | Detects statistically large swings in price                                                                                                                     |

Note that these all use forward-looking data to generate events (labels) that can be used to train a neural network.
They are omly used for training the models, they are not used in predicting events for live data.

## Classifier Types

There are many different types of neural network architectures that can be applied. Implementations are collected in
NNTClassifier.py.

The classifier is trained using the future-looking training signals. The neural network hopefully detects patterns based
on conditions at each hold/buy/sell signal, and can then apply that to predict buys/sells for live data.

Look at _NNTClassifier.ClassifierType_ for a current list of available classifier types

| NNTClassiferType  | Description                            |
|-------------------|----------------------------------------|
| Attention         | self-Attention (Transformer Attention) |
| AdditiveAttention | Additive-Attention                     |
| CNN               | Convolutional Neural Network           |
| Ensemble          | Ensemble/Stack of several Classifiers  |
| GRU               | Gated Recurrent Unit                   |
| LSTM              | Long-Short Term Memory (basic)         |
| LSTM2             | Two-tier LSTM                          |
| LSTM3             | Convolutional/LSTM Combo               |
| MLP               | Multi-Layer Perceptron                 |
| Multihead         | Multihead Self-Attention               |
| TCN               | Temporal Convolutional Network         |
| Transformer       | Transformer                            |
| Wavenet           | Simplified Wavenet                     |
| Wavenet2          | Full Wavenet                           |

## Testing a Strategy

To Do

## Training a Model

To Do

## HyperOpt

To Do

## Plotting Results

To Do

## Main Control Flags

To Do

## References

### Technical Indicators

Investopedia is a reliable and comprehensive website that explains different types of technical indicators for stock
trading. It provides detailed articles, tutorials, and examples on a wide range of financial topics, including technical
analysis and indicators. You can access their educational content on technical indicators
at: https://www.investopedia.com/terms/t/technicalindicator.asp

Investopedia covers popular technical indicators such as moving averages, relative strength index (RSI), stochastic
oscillator, Bollinger Bands, MACD (Moving Average Convergence Divergence), and many more. Each indicator is explained in
detail, including its calculation, interpretation, and how it can be used in stock trading strategies.

Additionally, Investopedia also offers a vast collection of articles on various aspects of investing, trading, and
finance, making it a valuable resource for individuals looking to expand their knowledge in these areas.

### Neural Networks (General)

There are several reference websites that provide introductions to different types of neural networks. Here are a few
recommended ones:

- TensorFlow's Neural Network Playground: TensorFlow's Neural Network Playground is an interactive website that allows you
to experiment with various neural network architectures. It provides a visual interface where you can create and train
different types of networks, such as feedforward networks, convolutional networks, and recurrent networks. The
playground also offers explanations and tutorials for each network type. You can access it
at: https://playground.tensorflow.org/

- DeepAI: DeepAI is an online platform that offers educational resources and articles on artificial intelligence and deep
learning. They have a section dedicated to neural networks, providing an introduction to different types of networks
such as feedforward, recurrent, convolutional, and generative networks. They also provide code examples and practical
explanations. You can explore their content at: https://deepai.org/machine-learning-glossary-and-terms/neural-network

- Stanford University CS231n: Stanford University's CS231n course on Convolutional Neural Networks for Visual Recognition
is freely available online. The course materials include lecture slides, lecture videos, and assignments. While the
focus is primarily on convolutional networks for image classification, the course also covers other neural network types
such as recurrent networks and generative models. You can find the course materials at: http://cs231n.stanford.edu/

- Towards Data Science: Towards Data Science is a popular online platform that covers a wide range of topics in data
science and machine learning. It features articles from industry professionals and researchers, many of which provide
introductions to different types of neural networks. You can explore their deep learning section for articles on
specific network architectures and their applications. The website is available at: https://towardsdatascience.com/

### Neural Networks (Specific Types)

#### (Self-) Attention
- Wikipedia: https://en.wikipedia.org/wiki/Attention_%28machine_learning%29
- Analytics India Magazine: https://analyticsindiamag.com/all-you-need-to-know-about-graph-attention-networks/
- Medium: https://yrajesh.medium.com/attention-in-neural-networks-d076b3e506d3
- Srome.github.io: http://srome.github.io/Understanding-Attention-in-Neural-Networks-Mathematically/
- Analytics India Magazine: https://analyticsindiamag.com/a-beginners-guide-to-using-attention-layer-in-neural-networks/

#### Convolutional Neural Network (CNN)
- Wikipedia: https://en.wikipedia.org/wiki/Convolutional_neural_network
- IBM: https://www.ibm.com/topics/convolutional-neural-networks
- TechTarget: https://www.techtarget.com/searchenterpriseai/definition/convolutional-neural-network
- DeepAI: https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network

#### Gated Recurrent Unit (GRU)
- Wikipedia: https://en.wikipedia.org/wiki/Gated_recurrent_unit
- DeepAI: https://deepai.org/machine-learning-glossary-and-terms/gated-recurrent-unit
- GeeksforGeeks: https://www.geeksforgeeks.org/gated-recurrent-unit-networks/
- Papers With Code: https://paperswithcode.com/method/gru
- Herong’s Tutorial Examples: https://www.herongyang.com/Neural-Network/RNN-What-Is-GRU.html

#### Long Short Term Memory (LSTM)
- Wikipedia: https://en.wikipedia.org/wiki/Long_short-term_memory
- Analytics India Magazine: https://analyticsindiamag.com/a-beginners-guide-to-long-short-term-memory-lstm/
- Towards Data Science: https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
- KDNuggets: https://www.kdnuggets.com/2021/05/long-short-term-memory-lstm-explained.html

#### Multi-Layer Perceptron (MLP)
- Wikipedia: https://en.wikipedia.org/wiki/Multilayer_perceptron
- Towards Data Science: https://towardsdatascience.com/multilayer-perceptron-explained-8498499781d5
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/multilayer-perceptron-mlp-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/multilayer-perceptron-explained.html

#### Multihead Attention
- Wikipedia: https://en.wikipedia.org/wiki/Multi-head_attention
- Towards Data Science: https://towardsdatascience.com/transformers-explained-multi-head-attention-deep-dive-into-heads-7fc3c3edd9c9
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/multi-head-attention-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/multi-head-attention-explained.html

#### Temporal Convolutional Network (TCN)
- Wikipedia: https://en.wikipedia.org/wiki/Temporal_convolutional_network
- Towards Data Science: https://towardsdatascience.com/temporal-convolutional-networks-tcn-explained-9d1b8f86a30
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/temporal-convolutional-networks-tcn-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/temporal-convolutional-networks-tcn-explained.html

#### Transformer
- Wikipedia: https://en.wikipedia.org/wiki/Transformer_(machine_learning_model
- Towards Data Science: https://towardsdatascience.com/transformers-explained-a-step-by-step-introduction-2a56e6dbae9f
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/transformers-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/transformers-explained.html

#### Wavenet
- Wikipedia: https://en.wikipedia.org/wiki/WaveNet
- Towards Data Science: https://towardsdatascience.com/wavenet-a-revolutionary-neural-network-for-speech-and-audio-processing-5d657b847f22
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/wavenet-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/wavenet-explained.html

