# Nexford-university-BAN6420-Module-6-Assignment-Fashion-MNIST-Classification

## GUIDE ON HOW THE CODE WORKS IN PYTHON
### STEP ONE: Import the necessary libraries
The libraries imported will handle the following
•	NumPy, a library for numerical operations, is especially useful for handling arrays and matrices.
•	Matplotlib's pyplot module for plotting graphs and visualizing data.
•	Loads the Fashion MNIST dataset, which contains 70,000 grayscale images of clothing items (10 categories like shirts, shoes, etc.)
•	Sequential model, which allows you to build a neural network layer-by-layer.
•	Import key layers used in CNNs:
o	Conv2D: Convolutional layer for feature extraction.
o	MaxPooling2D: Downsamples feature maps to reduce dimensionality.
o	Flatten: Converts 2D feature maps into 1D vectors.
o	Dense: Fully connected layer for classification.
o	Dropout: Regularization technique to prevent overfitting.
•	Converts class labels (integers) into one-hot encoded vectors, which are needed for classification tasks.
### STEP TWO: Load and Preprocess Data
This step handles the loading, preparing, and preprocessing of the Fashion MNIST dataset. The following is archived here.
Loading the Dataset
•	Loads the Fashion MNIST dataset from Keras.
•	x_train and x_test: Arrays of grayscale images (28×28 pixels).
•	y_train and y_test: Corresponding labels (integers from 0 to 9, each representing a clothing category).
Normalizing Pixel Values
•	Pixel values originally range from 0 to 255.
•	Dividing by 255 scales them to the range [0, 1], which helps the neural network train more efficiently and converge faster.
Reshaping for CNN Input
•	CNNs expect input in the format: (samples, height, width, channels).
•	-1 lets NumPy automatically calculate the number of samples.
•	28, 28: Image dimensions.
•	1: Number of channels (grayscale = 1; RGB would be 3).
So now the shape becomes:
•	x_train: (60000, 28, 28, 1)
•	x_test: (10000, 28, 28, 1)

### Step Three: Build the Convolutional Neural Network (CNN) model using Keras' Sequential
•	Sequential means the model is built layer-by-layer, in a linear stack.
•	Each layer feeds directly into the next.
### STEP FOUR: The CNN model gets compiled and trained
x_train, y_train_cat
•	Input images and one-hot encoded labels for training.
epochs=5
•	The model will go through the entire training dataset 5 times.
•	More epochs can improve accuracy but may also lead to overfitting if too many.
validation_data=(x_test, y_test_cat)
•	Uses test data to evaluate the model after each epoch.
•	Helps monitor performance on unseen data and detect overfitting early.
### STEP FIVE: This last step visualizes the model's predictions on sample test images
Class Labels
•	Maps numeric labels (0–9) to human-readable class names.
•	These correspond to the 10 categories in the Fashion MNIST dataset.
Select Sample Images
•	Picks the first two images from the test set.
•	sample_images: Image data.
•	sample_labels: True class labels (as integers).
Make Predictions
•	Uses the trained model to predict the class probabilities for each image.
•	Predictions[i] is a vector of 10 probabilities (one for each class).
•	np.argmax(predictions[i]) gives the index of the highest probability → predicted class.
Display Predictions with Images
•	Loops through the two sample images.
•	plt.imshow(...): Displays the image in grayscale.
•	reshape(28, 28): Converts image back to 2D for display.
•	plt.title(...): Shows the true label and the predicted label.
•	plt.axis('off'): Hides the axis for cleaner visuals.
•	plt.show(): Renders the image.

At the end the goal of making predictions for at least two images from the dataset was archived.

## GUIDE ON HOW THE CODE WORKS IN R
### STEP ONE: Load required libraries
•	Loads the Keras package for R.
•	For building and training the CNN.
•	For plotting and visualizing the images with labels.
### STEP TWO: Load Fashion MNIST Dataset
•	Loads the dataset of 28×28 grayscale images of clothing.
•	Splits it into training and test sets.
•	train_images and test_images: image data.
•	train_labels and test_labels: numeric labels (0–9) representing clothing categories.
### STEP THREE: Preprocess Images
•	Reshapes each image to include a channel dimension (needed for CNNs).
o	From (28, 28) → (28, 28, 1)
•	Divides pixel values by 255 to normalize them to the range [0, 1].
### STEP FOUR: One-Hot Encode Labels
•	Converts numeric labels (e.g., 3) into one-hot vectors (e.g., [0,0,0,1,0,0,0,0,0,0]).
•	Required for classification with categorical_crossentropy loss.
### STEP FIVE: Define Class Labels
•	Maps numeric labels (0–9) to human-readable clothing names.
### STEP SIX: Build CNN Model with 6 Layers
•	Layer 1: Conv2D with 32 filters and ReLU activation.
•	Layer 2: MaxPooling to reduce spatial size.
•	Layer 3: Conv2D with 64 filters.
•	Layer 4: Another MaxPooling layer.
•	Layer 5: Flatten to convert 2D data to 1D.
•	Layer 6: Dense layer with 128 neurons and ReLU.
•	Output Layer: Dense layer with 10 neurons (one per class) and softmax activation for probabilities.
### STEP SEVEN: Compile Model
•	Optimizer: Adam (adaptive learning rate).
•	Loss: Categorical crossentropy (for multi-class classification).
•	Metric: Accuracy.
### OTHER STEPS:
Train Model
•	Trains the model for 5 epochs.
•	Uses 64 images per batch.
•	Reserves 20% of training data for validation.
Evaluate Model
•	Tests the model on unseen data.
•	Prints the accuracy on the test set.
Predict First Two Test Images
•	Predicts probabilities for the first two test images.
•	Converts probabilities to class indices using which.max.
•	Subtracts 1 because R indices start at 1, but labels start at 0.
Visualize Predictions
•	Sets up a new plotting page with a 1-row, 2-column layout.
Image 1
•	Plots the first image.
•	Displays its true and predicted label.
Image 2
•	Same as above, but for the second image.
At the end the goal of making predictions for at least two images from the dataset was archived.









