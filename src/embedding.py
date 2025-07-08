from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding, PermutationInvariantEmbedding
import torch.nn as nn

def get_embedding(embedding_type="identity"):
    if embedding_type == "identity":
        return nn.Identity()
    if embedding_type == "FCE":
        return FCEmbedding(input_dim=2551)
    elif embedding_type == "CNN":
        return CNNEmbedding(input_shape=(2551,))


