# 🤖 Neural Network From Scratch: `anndy`  

A painfully simple autograd engine built in vanilla Python, no PyTorch, no numpy, just math. 

--- 

## 🐍 Try It Out
Neural network utility classes and functions in `anndy.py`.  
- Dependencies: none.

For a simple demo on a regression task, run `demo.py`.  
- Dependencies: `matplotlib`, `tqdm`

---

## 🧠 Example Usage
Initialization: 
```py
nn = anndy.MLP((8, "tanh"), (4, "tanh"), (2, "relu"), (1, "relu"))  # Initialize multi-layer perceptron
```
Optimization loop: 
```py
preds = [nn(i) for i in data_y]  # Forward pass
loss = anndy.mean_squared_error(data_y, preds)  # Compute loss

nn.zero_grad()  # Reset gradients
loss.backward()  # Backward pass
nn.nudge(0.001)  # Descend gradient
```

---

## 🔗 Links & Sources
- Based on Andrej Karpathy's video: https://youtu.be/VMj-3S1tku0 (recommend building this project if you're just getting started with neural networks)
- Concrete strength dataset used in the demo: https://www.kaggle.com/datasets/niteshyadav3103/concrete-compressive-strength
- My explanation article: https://medium.com/@andyyy.yuuu/neural-network-from-scratch-no-pytorch-no-numpy-just-vanilla-python-56be311eac89
- My jazz playlist (entirely unrelated): https://open.spotify.com/playlist/6Uw2DCToKXVVqmnu33m8Xy
