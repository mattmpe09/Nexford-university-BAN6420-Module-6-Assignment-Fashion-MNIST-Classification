# Load required libraries
library(keras)
library(grid)
library(gridExtra)

# Load Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Preprocess images
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1)) / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1)) / 255

# One-hot encode labels
train_labels <- to_categorical(train_labels, 10)
test_labels <- to_categorical(test_labels, 10)

# Define class labels
labels <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

# Build CNN model with 6 layers
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Train model
model %>% fit(
  train_images, train_labels,
  epochs = 5,
  batch_size = 64,
  validation_split = 0.2
)

# Evaluate model
score <- model %>% evaluate(test_images, test_labels)
cat('Test accuracy:', score$accuracy, '\n')

# Predict first two test images
predictions <- model %>% predict(test_images[1:2,,, drop = FALSE])
predicted_classes <- apply(predictions, 1, which.max) - 1
true_classes <- apply(test_labels[1:2,], 1, which.max) - 1

# Visualize predictions
grid.newpage()
pushViewport(viewport(layout = grid.layout(1, 2)))

# Image 1
print(
  grid.raster(matrix(test_images[1,,,1], 28, 28)[,28:1], interpolate = FALSE),
  vp = viewport(layout.pos.row = 1, layout.pos.col = 1)
)
grid.text(paste0("True: ", labels[true_classes[1] + 1],
                 "\nPredicted: ", labels[predicted_classes[1] + 1]),
          x = 0.25, y = 0.95, gp = gpar(fontsize = 12))

# Image 2
print(
  grid.raster(matrix(test_images[2,,,1], 28, 28)[,28:1], interpolate = FALSE),
  vp = viewport(layout.pos.row = 1, layout.pos.col = 2)
)
grid.text(paste0("True: ", labels[true_classes[2] + 1],
                 "\nPredicted: ", labels[predicted_classes[2] + 1]),
          x = 0.75, y = 0.95, gp = gpar(fontsize = 12))
