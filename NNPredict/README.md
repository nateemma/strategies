# Neural Network Prediction (NNPredict)

(still a work in progress)

As the name suggests, these strategies implement neural network algorithms that predict future gains. 

In general, the flow is:

1. add 'standard' technical indicators to the dataframe
2. add 'hidden' and forward-looking indicators to a copy of the dataframe.<br>Hidden indicators are those that are not
   explicitly forward looking, but might accidentally do so, e.g. by using dataframe-wide averages.<br> Forward-looking
   indicators do explicitly look forward in the data, which is OK when used for training the neural network, but not for
   normal operation.
4. if a model exists for the neural network then load it, otherwise train a new model.<br> Training involves using the
   forward-looking data to identify the actual gains, then feeding those to the neural network along with
   the dataframe (without the hidden and forward-looking data). The neural network then iterates over the data until it
   finds a good set of weights such that the outputs of the model match the supplied gains as closely as possible.<br>
   The resulting model is then saved for future use.
5. each time new data is received, it is fed through the model, which predicts future gain, which are then
   added to the dataframe
6. populate_entry_trend() and populate_exit_trend() can then use the predicted gain column to add
   entry/exit conditions the dataframe, as normal

Note that, contrary to how the media portrays Machine Learning, the neural network algorithms are actually not that
great at predicting - if they were, I would be driving a lambo by now!<br>
Time Series forecasting is actually a leading edge problem in machine learning, and it doesn't get anywhere near as much
attention as natural language processing (NLP)

## Main Classes and Conventions

| Class                                      | Description                                                                                                         |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| NNPredict                                       | This is the base class that implements the general flow                                                             |
| NNPredict_<_predictor\_type_> | This is a strategy that uses a _predictor\_type_ (Neural Network) |                                      |
| NNTPredictor_<_predictor\_type_>              | Implementations of various types of neural network models                                                           |


The actual strategies are fairly simple, and essentially just set variables that tell the NNPredict base class which options
to use.

## Classifier Types

There are many different types of neural network architectures that can be applied. Implementations are collected in
NNPredictlassifier.py.

The classifier is trained using the future-looking training signals. The neural network hopefully detects patterns based
on conditions at each hold/buy/sell signal, and can then apply that to predict buys/sells for live data.

Look at _NNPredictlassifier.ClassifierType_ for a current list of available classifier types

| NNPredictlassiferType  | Description                            |
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

### About Models

The NNPredict strategies use neural networks that are implemented using keras and Tensorflow. Tensorflow is the framework
provided by Google for working with neural networks. keras is an API that simplifies the specification of network layers
and management of models.

Models are located in the directory _user\_data/strategies/NNPredict/models_
in a subdirectory that matches the name of the strategy.


**If you change the implementation of a model, you must retrain all models that use that architecture**

## Testing a Strategy

If you have the zsh shell, then you can use the helper scripts, which are located in user_data/strategies/scripts

### Examples:

Test strategy over the last 30 days

```commandline
zsh user_data/strategies/scripts/test_strat.sh -n 30 NNPredict NNPredict_Transformer
```

Test all NNPredict strategies that use the LSTM model
(Note that you need the quotes)

```commandline
zsh user_data/strategies/scripts/test_group.sh -n 30 NNPredict "NNPredict_LSTM*"
```

## Training a Model

Training is where you feed (lots of) data into an empty neural network model so that it 'learns' the relationship
between the input data and the training signals (hold/buy/sell).

Currently, the strategy will automatically go into training mode if the model is not present in the file system.
So, the 'easy' way to train a model is to delete the model file(s)

```commandline
# delete model file(s)
rm -r user_data/strategies/NNPredict/models/NNPredict_Transformer

# train the model
zsh user_data/strategies/scripts/test_strat.sh -n 360 NNPredict NNPredict_Transformer
```

Note that you can also use the "_-m_" option of test_group.sh to re-train missing models:

```
# train any missing models that use Transformer:
zsh user_data/strategies/scripts/test_group.sh -n 360 -m NNPredict "NNPredict_*Transformer"

```

## HyperOpt

```commandline
zsh user_data/strategies/scripts/hyp_group.sh -n 90 -e 100 -l CalmarHyperOptLoss NNPredict "NNPredict_macd*"
```


## Plotting Results

```commandline
freqtrade plot-dataframe --strategy-path user_data/strategies/NNPredict -c user_data/strategies/NNPredict/config_NNPredict.json  -p SOL/USD --timerange=20230301-20230307 -s NNPredict_macd_Transformer
```

and display the resulting html file in your browser

If you look in the strategy file (just below the class declaration, you will se a definition of the variable
_plot_config_). This controls what is displayed on the plot, and the colours used.

For example, in _NNPredict.py_, this is:

```commandline
    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
        },
        'subplots': {
            "Diff": {
                'predicted_gain': {'color': 'purple'},
                'gain': {'color': 'orange'},
                '%future_gain': {'color': 'lightblue'},
                'target_profit': {'color': 'lightgreen'},
                'target_loss': {'color': 'lightsalmon'}
            },
        }
    }
```

FYI, the colours are the standard ones supported by all browsers, and you can find a display of the
colours [here](https://www.w3schools.com/cssref/css_colors.php)

Anything prefixed with "%" means that it is a 'hidden' indicator (not visible to the model). 

The plot function is quite useful for checking on what signals are generated for training, and what signals the
model actually provides. On the plot, you can hide/display indicators by clicking on the name in key list (on the
right). If you hide all of the indicators except %train_buy and predict_buy then you can compare the training signal (
%train_buy) and the signals generated by the model (predict_buy).


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

- TensorFlow's Neural Network Playground: TensorFlow's Neural Network Playground is an interactive website that allows
  you
  to experiment with various neural network architectures. It provides a visual interface where you can create and train
  different types of networks, such as feedforward networks, convolutional networks, and recurrent networks. The
  playground also offers explanations and tutorials for each network type. You can access it
  at: https://playground.tensorflow.org/

- DeepAI: DeepAI is an online platform that offers educational resources and articles on artificial intelligence and
  deep
  learning. They have a section dedicated to neural networks, providing an introduction to different types of networks
  such as feedforward, recurrent, convolutional, and generative networks. They also provide code examples and practical
  explanations. You can explore their content at: https://deepai.org/machine-learning-glossary-and-terms/neural-network

- Stanford University CS231n: Stanford University's CS231n course on Convolutional Neural Networks for Visual
  Recognition
  is freely available online. The course materials include lecture slides, lecture videos, and assignments. While the
  focus is primarily on convolutional networks for image classification, the course also covers other neural network
  types
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
- Towards Data
  Science: https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
- KDNuggets: https://www.kdnuggets.com/2021/05/long-short-term-memory-lstm-explained.html

#### Multi-Layer Perceptron (MLP)

- Wikipedia: https://en.wikipedia.org/wiki/Multilayer_perceptron
- Towards Data Science: https://towardsdatascience.com/multilayer-perceptron-explained-8498499781d5
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/multilayer-perceptron-mlp-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/multilayer-perceptron-explained.html

#### Multihead Attention

- Wikipedia: https://en.wikipedia.org/wiki/Multi-head_attention
- Towards Data
  Science: https://towardsdatascience.com/transformers-explained-multi-head-attention-deep-dive-into-heads-7fc3c3edd9c9
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
- Towards Data
  Science: https://towardsdatascience.com/wavenet-a-revolutionary-neural-network-for-speech-and-audio-processing-5d657b847f22
- Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/wavenet-explained/
- KDNuggets: https://www.kdnuggets.com/2021/05/wavenet-explained.html

