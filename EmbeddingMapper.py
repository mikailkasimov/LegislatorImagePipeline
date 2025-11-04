import numpy as np
import matplotlib.pyplot as plt
from cuml.manifold import TSNE  # Import from cuml instead of sklearn
from cuml.manifold import UMAP  # Import from cuml instead of umap

class EmbeddingMapper:
    def __init__(self, dimension=512, components=3, tsne = None,umap = None):
        self.dimension=dimension    #Dimension of input embeddings
        self.components=components  #Dimension to reduce embeddings to (2D or 3D)
        if tsne is not None:
            self.tsne = tsne
        else:
            self.tsne = TSNE(
                n_components=components,
                perplexity=30,
                learning_rate=200,
                init="pca",
            )
        if umap is not None:
            self.umap = umap
        else:
            self.umap = UMAP(
                n_components=components,
                n_neighbors=15,
                min_dist=0.1,
            )


    """
    Transform embeddings to lower dimension via using t-SNE

    param: embeddings_list: list of 'd' dimension embeddings
    return: tsne_embeddings: N x D darray of reduced dimension embeddings (specified by 'components' arg)
    """
    def tsne_transform(self, embeddings_list):
        return self.tsne.fit_transform(embeddings_list)
    

    """
    Transform embeddings to lower dimension via using UMAP

    param: embeddings_list: list of 'd' dimension embeddings
    return: tsne_embeddings: N x D ndarray of reduced dimension embeddings (specified by 'components' arg)
    """
    def umap_transform(self, embeddings_list):
        return self.umap.fit_transform(embeddings_list)
    


    """
    Transforms original embeddings and plots them with t-SNE.

    param: embeddings_list: list of 'd' dimension embeddings.
    param: labels (optional): A list of labels for color-coding the points.
    return: None
    """
    def plot_tsne(self, embeddings_list, labels=None):
        print("Running t-SNE transformation...")
        tsne_embeddings = self.tsne_transform(np.array(embeddings_list))
        if self.components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='viridis')
            plt.title('2D t-SNE')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True)
            plt.show()
        elif self.components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], c=labels, cmap='viridis')
            ax.set_title('3D t-SNE')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.show()
        else:
            raise ValueError("Plotting is only supported for 2 or 3 components.")



    """
    Transforms original embeddings and plots them with UMAP.

    param: embeddings_list: list of 'd' dimension embeddings.
    param: labels (optional): 1 x N*D ndarray of labels for color-coding the points.
    """
    def plot_umap(self, embeddings_list, labels=None):
        print("Running UMAP transformation...")
        umap_embeddings = self.umap_transform(np.array(embeddings_list))
        
        if self.components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis')
            plt.title('2D UMAP Visualization')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True)
            plt.show()
        elif self.components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], c=labels, cmap='viridis')
            ax.set_title('3D UMAP Visualization')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.show()
        else:
            raise ValueError("Plotting is only supported for 2 or 3 components.")