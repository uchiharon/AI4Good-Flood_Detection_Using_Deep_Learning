conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0


TensorFlow 2 Setup Link
https://www.tensorflow.org/install/


1. create virtual environment for python 3.10
conda create -n py310 python=3.10
2. activate the environment
conda activate py310
3. Install cuda on the environment
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

4. Install tensorflow 2.10

python -m pip install "tensorflow<2.11"

Test GPU
import tensorflow as tf
tf.config.list_physical_device('GPU')
tf.test.is_gpu_available()