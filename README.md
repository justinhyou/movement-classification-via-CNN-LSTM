# movement-classification-via-CNN-LSTM
Classification of body movement (from UCI database) by neural networks, building upon other classification networks and introducing an optimizing, convolutional neural network that significantly increased accuracy.

Originally run on:
Ubuntu 14.04 LTS 

Dependencies:
Program specifics:
Python 2.7.6 GCC 4.8.4
Tensorflow 0.12.1
Numpy 1.110
Scipy 0.13.3
Sci-kit learn 0.18.1
Matplotlib 1.3.1

Optimization:
CUDA toolkit 8.0 and CuDNN v5

Running code "as is": Run "python edit.py"

https://www.cs.hmc.edu/~hyou/

Project Description:

Surrogate robotics requires an interface between human body movement and robotic interpretation with high accuracy and speed. In search of a highly responsive interface, I have developed a method to improve existing neural networks. Not yet tested to completion, the abstraction layer by convolutional neural network superimposed on the layered, Long Short-Term Memory (LSTM) classifier network may provide a method for accuracy improvement without a significant loss in reactivity (during classification, as well as training). The abstraction allows the same generic movements (i.e. walking up the stairs) to become quickly learned and classified with accuracy, the network learning to adapt to a specific user and subject. Preliminary results with elementary data have shown improved accuracy in body movement classification.

Using primitive data measurements via two instruments on a phone (accelerometer and gyroscope), six distinct body movements were classified with high accuracy. Beginning with two LSTM neural network layers, the accuracy was 90% following 300 iterations over the data set (70% training, 30% validation). With the addition of a small Convolutional Neural Network (CNN) above these layers, the accuracy was increased to 94% following the exact same training parameters (epoch-limited, with the same training optimization techniques). There was no noticeable increase in time with a single four-core CPU.

Future applications of this research include applying abstraction of neural networks in Body Machine Interfaces (BMI), thereby providing a mechanism for quick adaptation and specialized training and learning for a specific individual. This is especially significant, given the brain is polymorphic: electroencephalogram (EEG) data or more specifically for medical application, deep electrode recordings, collected for two individuals completing the same physical task is significantly different even over just the standardized motor cortex. This is especially true for those born without limbs or those who have lost limbs. Due to the brain's neuroplasticity, motor cortex areas form and reform, respectively, such that body movement and the corresponding subsection of the cortex cannot be easily predicted or generalized. To overcome the issue in the medical context, such a neural network with applied abstraction can shorten the difficult training period and arduous physical therapy, decreasing costs for the life-enhancing procedure while delivering a more accurate physical exhibition of mental intent.

For questions, please contact (Justin) Hyobin You at hy002014@mymail.pomona.edu.
