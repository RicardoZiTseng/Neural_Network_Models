###Implement LeNet-5 with tensorflow

Because we use the mnist dataset to test the LeNet model, so the dimension of input data is **32 by 32 by 1**

The structure of LeNet-5 shows here:
$$
32 \times 32 \times 1 \rightarrow_{conv, \ 5\times5, strides = 1} 28\times28\times6 \rightarrow_{\text{avg pool}, f = 2, s = 2} \\
14\times14\times6 \rightarrow_{\text{conv}, \ 5\times 5, strides = 1}10\times10\times16 \rightarrow_{\text{avg pool}, f=2, s=2}\\
5\times5\times16 \rightarrow \text{Flatten, dimension = 120} \rightarrow_{\text{full connected}} \text{Full Connect, dimension = 84}\\
\rightarrow \text{Softmax, dimension = 10}
$$
