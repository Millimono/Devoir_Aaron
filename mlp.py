import torch
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
            super(Linear, self).__init__()

             # Initialisation de He (Kaiming Normal)
                #std = (2 / in_features) ** 0.5  # écart-type selon He
                #self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
                # self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2 / in_features) ** 0.5)

            self.weight = nn.Parameter(torch.randn(out_features, in_features) )
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self.weight.T + self.bias



import torch
from torch import nn
from typing import List, Tuple

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 0, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"

        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)

        # Initialisation des poids et biais
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)



    def _build_layers(self, input_size: int, hidden_sizes: List[int], num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """Construit les couches cachées et la couche de sortie avec ta classe Linear"""

        layers = []
        prev_size = input_size

        # Construire les couches cachées
        for hidden_size in hidden_sizes:
            layers.append(Linear(prev_size, hidden_size))
            prev_size = hidden_size  # Mise à jour pour la prochaine couche

        hidden_layers = nn.ModuleList(layers)

        # Construire la couche de sortie
        output_layer = Linear(prev_size, num_classes)

        return hidden_layers, output_layer


    def activation_fn(self, activation: str, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applique la fonction d'activation choisie
        """
        if activation == "relu":
            return inputs * (inputs > 0)  # ReLU : max(0, x)
        elif activation == "tanh":
            return (torch.exp(inputs) - torch.exp(-inputs)) / (torch.exp(inputs) + torch.exp(-inputs))  # tanh(x)
        elif activation == "sigmoid":
            return 1 / (1 + torch.exp(-inputs))  # Sigmoid(x)
        else:
            raise ValueError("Activation function not supported")


    def _initialize_linear_layer(self, module: nn.Linear) -> None:
            """
            Initialisation manuelle :
            - Poids selon Xavier Normal (Glorot Normal)
            - Biais mis à zéro
            """
            in_features = module.weight.shape[1]  # Nombre de neurones en entrée
            out_features = module.weight.shape[0]  # Nombre de neurones en sortie
            std = (2 / (in_features + out_features)) ** 0.5  # Xavier Normal

            with torch.no_grad():  # Désactive le calcul du gradient pour l'initialisation
                module.weight.copy_(torch.randn(out_features, in_features) * std)  # Poids Xavier Normal
                module.bias.zero_()  # Biais à zéro


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant des images à travers le réseau MLP
        """
        x = images.view(images.size(0), -1)  # Aplatir l'image en un vecteur

        for layer in self.hidden_layers:
            x = self.activation_fn(self.activation, layer(x))

        logits = self.output_layer(x)
        return logits
