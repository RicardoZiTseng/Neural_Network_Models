### Implement LeNet-5 with tensorflow

Because we use the mnist dataset to test the LeNet model, so the dimension of input data is **32 by 32 by 1**

The structure of LeNet-5 shows here:
![](https://github.com/RicardoZiTseng/Neural_Network_Models/blob/master/LeNet-5/Pictures/LeNet-5_structure.png)

And the performance of LeNet-5 shows very well!
Here is the result:

| The number of test dataset | test accuracy |
| :------------------------: | :-----------: |
|            256             |     98.4%     |
|            512             |     98.6%     |
|            1024            |     97.8%     |

And the loss during the training shows here:

when the learning rate is **0.0003**:

![](https://github.com/RicardoZiTseng/Neural_Network_Models/blob/master/LeNet-5/Pictures/learning_rate_2.jpg)

when the learning rate is **0.001**:

![](https://github.com/RicardoZiTseng/Neural_Network_Models/blob/master/LeNet-5/Pictures/learning_rate_3.jpg)

when the learning rate is **0.009**:

![](https://github.com/RicardoZiTseng/Neural_Network_Models/blob/master/LeNet-5/Pictures/learning_rate_2.jpg)
