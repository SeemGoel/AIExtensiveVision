Ask:

The assignment is to train a NN from scratch. Creating a simple network to show different steps of training including back propagation.
This excels sheet take us through step by step forward propagation and back propagation steps of the network.

Assumptions:
- Two Input neurons (i1, i2)
- One hidden layer of size 2 (h1, h2)
- Activation function on hidden layers (out_h1, out_h2)
- One output layer (O1, O2)
- Activation Function on Output (out_O1, out_O2)
- Target (t1, t2)
- Error Total (Total = E1 + E2)
- Weights (w1,w2 ………, w8)


### Forward Propagation
- h1=w1i1+w2i2
- out_h1 = σ(h1) = σ(w1i1+w2i2)
- h2=w3i1+w4i2
- out_h2 = σ(h2) = σ(w3i1+w4i2)
- o1 = w5*out_h1 + w6 * out_h2
- out_o1 = σ(o1) = σ(w5*out_h1 + w6 * out_h2)
- o2 = w7*out_h1 + w8 * out_h2
- out_o2 = σ(o2) = σ(w7*out_h1 + w8 * out_h2)
- E_Total = E1 + E2
- E1 = 1/2 *(t1 - out_o1) **2
- E2 = 1/2 *(t2 - out_o2) **2




