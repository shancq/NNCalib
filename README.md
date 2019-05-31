# NNCalib

Tensorflow model code for **Diagnosis of Calibration State for Massive Antenna
Array via Deep Learning**

## Example
```python
N = 64  # length of a source signal vector 
M = N // 8  # length of a observation vector 
dilated_layers = 6
network = ConveCsNet(num_of_blocks=dilated_layers, comp_dims=M * 2, ori_dims=N * 2, num_filters=4, k=3)
#x = data_loader()  generate the sparse input x
y, _ = network(x)
```
