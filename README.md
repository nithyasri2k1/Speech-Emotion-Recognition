# Speech-Emotion-Recognition
Speech Emotion Detection (SED) is an important research area in the field of 
speech processing and human-computer interaction. SED involves the 
recognition of emotions expressed by a speaker in a given speech signal. The 
ability to detect emotions can have various applications, such as in speech 
therapy, human-robot interaction, and sentiment analysis. In recent years, 
there has been a growing interest in developing efficient methods for SED 
using machine learning techniques. One of the challenges in SED is the 
extraction of relevant features from the audio dataset that can capture the 
emotional content of speech. Traditional approaches for feature extraction 
include using Mel Frequency Cepstral Coefficients (MFCCs), which have been 
widely used in speech processing. In this paper, we propose a novel approach 
for SED using PySpark in Multilayer Perceptron (MLP) with an audio dataset. 
PySpark is a powerful distributed computing framework that can handle largescale datasets. The proposed approach involves the extraction of relevant 
features from the audio dataset, such as MFCCs and Energy, followed by the 
application of a neural network-based MLP model for classification. The use of 
PySpark allows for the processing of large-scale audio datasets, which is 
essential for achieving high accuracy in SED.
The goal of speech emotion recognition (SER) is to infer the speaker’s 
emotional state from the speech signal. SER is a multi-step procedure that 
includes phases like feature extraction, feature selection, and classification.
Deep learning-based methods have produced encouraging outcomes in SER 
in recent years. A popular feed forward neural network for classification 
problems is the multi-layer perceptron (MLP). The weights of the network are
trained usingbackpropagation in MLP, which has multiple layers of nodes. The 
capability of MLP to learn intricate nonlinear correlations between input and 
output is well recognised. We provide an MLP-based SER model for feature 
extraction and classification in this study. Speech is a crucial component of
humancommunication, and speech is largely influenced by emotions The 
capability of MLP to learn intricate nonlinear correlations between input and 
output is well recognised. We provide an MLP-based SER model for feature
extraction and classification in thisstudy.


After training the MLP model using PySpark's MLlib library, we achieved an 
accuracy of 85% and an F1 score of 0.64on the testing set. These results 
indicate that the MLP model is able to classify emotions with high accuracy.
The proposed system's performance is quite good as it achieved an accuracy of 
85% and an F1 score of 0.64 on the testing set. The accuracy and F1 score of 
the system can be improved by using a larger and more diverse dataset and by 
fine-tuning the model's hyperparameters. Additionally, the system's 
performance can be further enhanced by incorporating other advanced 
machine learning algorithms such as Convolutional Neural Networks (CNN) or 
Recurrent Neural Networks (RNN).
One of the significant advantages of using PySpark for speech emotion 
detection is the ability to process and analyze a large amount of audio data in 
parallel. PySpark's MLlib library also provides a range of tools and algorithms 
for feature extraction, model training, and evaluation, making it easier to 
develop and deploy machine learning models.
Overall, the proposed system using MLP in PySpark for speech emotion 
detection has shown promising results, and further improvements can be 
made by incorporating advanced algorithms and fine-tuning the 
hyperparameters.

