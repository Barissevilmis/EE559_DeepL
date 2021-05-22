import json
from models import Linear, Sequential
from activations import ReLU, Tanh, Sigmoid
from losses import MSE
from utils import generate_set, hyperparameter_tuning


train_input, train_target, test_input, test_target = generate_set(1000)


# A distinct model for every different activation function
relu_model = Sequential(Linear(2, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 1), Sigmoid())

tanh_model = Sequential(Linear(2, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 1), Sigmoid())

sigmoid_model = Sequential(Linear(2, 25), Sigmoid(),
                           Linear(25, 25), Sigmoid(),
                           Linear(25, 25), Sigmoid(),
                           Linear(25, 1), Sigmoid())


model_params_sgd = {'lr': [1e-1]}


best_param_sgd_relu, train_acc, val_acc = hyperparameter_tuning(relu_model, optimizer="sgd", criterion=MSE(
), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

print(train_acc)
print(val_acc)

exit(0)

with open('best_param_sgd_relu.txt', 'w') as file:
    json.dump(best_param_sgd_relu, file)

best_param_sgd_tanh = hyperparameter_tuning(tanh_model, optimizer="sgd", criterion=MSE(
), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

with open('best_param_sgd_tanh.txt', 'w') as file:
    json.dump(best_param_sgd_tanh, file)

best_param_sgd_sigmoid = hyperparameter_tuning(sigmoid_model,  optimizer="sgd", criterion=MSE(
), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

with open('best_param_sgd_sigmoid.txt', 'w') as file:
    json.dump(best_param_sgd_sigmoid, file)
