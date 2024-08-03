import torch
import matplotlib.pyplot as plt

def trainModel(model, train_dataloader, test_dataloader, criterion, optimiser, num_epochs, early_stopping = 3, verbose=False, return_model=False):
  """
  Function to train a NN model.
  Args:
    model (nn.Module): The model to train.
    train_dataloader (DataLoader): The training data loader.
    test_dataloader (DataLoader): The test data loader.
    criterion (nn.Module): The loss function.
    optimiser (torch.optim): The optimiser.
    num_epochs (int): The number of epochs to train for.
    early_stopping (int): The number of epochs to wait before early stopping.
  Returns:
    avg_train_loss (float): The average training loss.
  """

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  best_loss = float('inf')
  best_model = None
  counter = 0

  for epoch in range(num_epochs):
    model.train() # set to training mode
    train_loss = 0
    for features, target in train_dataloader:
      features, target = features.to(device), target.to(device) # move data to GPU

      optimiser.zero_grad() # zero parameter gradients
      outputs = model(features) # forward pass (model calculates output based on input)
      loss = criterion(outputs, target) # comput loss
      loss.backward() # backward pass (backpropogation)
      optimiser.step() # update model parameters using gradient calculated in backpropogation
      train_loss += loss.item()

      avg_train_loss = train_loss / len(train_dataloader)
      if verbose:
        print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss:.3f}')

    # test loss
    model.eval() # set to evaluation mode
    test_loss = 0
    with torch.no_grad(): # disable gradient calculation
      for features, target in test_dataloader:
        features, target = features.to(device), target.to(device) # move data to GPU

        outputs = model(features)
        loss = criterion(outputs, target)
        test_loss += loss.item()

      avg_test_loss = test_loss / len(test_dataloader)

    # Early stopping
    if avg_test_loss < best_loss:
      best_loss = avg_test_loss
      best_model = model
      counter = 0
    else:
      counter += 1
      if counter >= early_stopping:
        if verbose:
          print(f'Early stopping at epoch {epoch+1}')
        break
    
  if return_model:
    return model, avg_train_loss     
  else:
    return avg_train_loss


def reinitialiseModel(model_class, model_params):
  return model_class(**model_params)


def validateModel(model, val_dataloader, criterion):
    model.eval()
    val_loss = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for features, target in val_dataloader:    
            features, target = features.to(device), target.to(device) # move data to GPU

            outputs = model(features)
            val_loss += criterion(outputs, target).item()

    avg_val_loss = val_loss / len(val_dataloader)

    return avg_val_loss
  

def findbestModel(model_class, model_params, train_dataloader, test_dataloader, val_dataloader, criterion, learning_rate, num_epochs, n_initialise=5):
  best_model = None
  best_val_loss = float('inf')
  best_train_loss = float('inf')

  for i in range(n_initialise):
    print(f'Initialising: {i}')
    model = reinitialiseModel(model_class, model_params)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, train_loss = trainModel(model, train_dataloader, test_dataloader, criterion, optimiser, num_epochs, return_model=True)
    val_loss = validateModel(model, val_dataloader=val_dataloader, criterion=criterion)  

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_train_loss = train_loss
      best_model = model
    
  print(f'Train loss: {best_train_loss:.3f} \nValidation loss: {best_val_loss:.3f}')
  return best_model    


def plotPredictions(y_test, y_pred):
  fig, ax = plt.subplots()
  ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')

def testFunction(x):
  print(x*2)