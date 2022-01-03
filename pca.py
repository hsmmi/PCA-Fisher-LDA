from matplotlib import pyplot as plt
import numpy as np
from dataset import Dataset
from copy import deepcopy


class PCA():
    def __init__(self):
        self.dataset = Dataset()
        self.number_of_component = None
        self.mean_feature = None
        self.centered_sample = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.transformed_sample = None
        self.reconstructed_sample = None
        self.number_of_visual_sample = None
        self.visualize_sample = None

    def fit(self, dataset: Dataset):
        mean_feature = np.mean(dataset.sample, axis=0)
        centered_sample = dataset.sample - mean_feature
        cov_mat = np.cov(centered_sample.T)

        # It's decreasing sorted by eigenvalues
        eigenvectors, eigenvalues, vh = np.linalg.svd(cov_mat)
        # Using transpose to each row became eigenvalues
        eigenvectors = eigenvectors.T

        # eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        # dec_sorted_index = np.flip(np.argsort(eigen_values))
        # eigenvector_subset = eigen_vectors[dec_sorted_index]

        self.dataset = dataset
        self.centered_sample = centered_sample
        self.mean_feature = mean_feature
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.number_of_visual_sample = 5
        np.random.seed(2)
        self.visualize_sample = np.random.randint(
            self.dataset.number_of_sample, size=self.number_of_visual_sample)

    def transformation(self, number_of_component=None):
        if number_of_component is None:
            number_of_component = self.number_of_component
        else:
            self.number_of_component = number_of_component
        eigenvector_subset = self.eigenvectors[:number_of_component]
        transformed_sample = self.centered_sample @ eigenvector_subset.T

        self.transformed_sample = transformed_sample

    def reconstruction(self):
        eigenvector_subset = self.eigenvectors[:self.number_of_component]
        reconstructed_sample = self.transformed_sample @ eigenvector_subset
        self.reconstructed_sample = reconstructed_sample

    def visualization(self, images_vector: np.ndarray, title: str = None,
                      random: bool = True, add_mean: bool = True):
        images_vector = deepcopy(images_vector)
        if add_mean is True:
            images_vector += self.mean_feature
        image_size = self.dataset.image_size
        images = images_vector.reshape((-1, image_size, image_size))
        fig = plt.figure()
        number_of_visual_sample = self.number_of_visual_sample
        if random is True:
            visualize_sample = self.visualize_sample
        else:
            visualize_sample = range(self.number_of_visual_sample)
        for i in range(number_of_visual_sample):
            ax = fig.add_subplot(1, number_of_visual_sample, i + 1)
            ax.imshow(images[visualize_sample[i]], cmap="gray")
            ax.set_title(f'image #{visualize_sample[i]}')
        fig.suptitle(title)
        fig.set_size_inches(3 * self.number_of_visual_sample, 3.5)
        fig.savefig(f'report/{title}.png', dpi=100)


dataset = Dataset()
dataset.read_dataset('dataset/jaffe', 64)
pca = PCA()
pca.fit(dataset)
diffrent_k = [1, 40, 120]
pca.visualization(pca.centered_sample, 'PCA-Images before transformation')
pca.visualization(pca.eigenvectors, 'PCA-Eigenfaces', False, False)
for k in diffrent_k:
    pca.transformation(k)
    pca.reconstruction()
    pca.visualization(
        pca.reconstructed_sample,
        f'PCA-Images after reconstruction with k = {k}')
