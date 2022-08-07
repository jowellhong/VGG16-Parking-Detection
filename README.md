# Detection of Parking Lot

This project intends to use deep learning to detect parking spaces. It mainly uses Convolutional Neural Network (CNN) to determine the occupancy of the parking spaces.

To replicate the model:
1. Download the PKlot dataset: http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
2. Load create_dataset_index.ipynb notebook to organise the pictures, mainly using segmented pictures for training. Modify necessary directories.
3. Load cnn_models_vgg16.ipynb notebook to start the training. Log on to Weight&Biases to save the training results.
4. To test the model, select a sample parking lot image. Use GIMP-2.10 to manually locate all coordinates for the parking spaces.
5. Parkinglot.jpeg is tested with main.py.
