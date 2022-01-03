from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from dataset import Dataset


class FisherLDA():
    def __init__(self):
        self.dataset = Dataset()
        self.number_of_component = None
        # It's c * n matrix that (i, j) is mean of
        # feature j of samples in class i
        self.mean_class_feature = None
        self.mean_overall_feature = None
        self.between_class_scatter = None
        self.within_class_scatter = None
        self.centered_sample = None
        self.eigenvectors = None
        self.transformed_sample = None
        self.reconstructed_sample = None
        self.number_of_visual_sample = None
        self.visualize_sample = None

    def fit(self, dataset: Dataset):
        mean_class_feature = np.array([
            np.mean(dataset.sample[(dataset.label == label).flatten()], axis=0)
            for label in dataset.diffrent_label
        ])
        mean_overall_feature = np.mean(dataset.sample, axis=0)
        number_of_feature = dataset.number_of_feature

        between_class_scatter = np.zeros(
            (number_of_feature, number_of_feature))
        for i in range(dataset.number_of_diffrent_label):
            size_of_class = dataset.count_diffrent_label[i]
            distance = (
                mean_class_feature[i] - mean_overall_feature).reshape(-1, 1)
            between_class_scatter += size_of_class * (distance@distance.T)

        within_class_scatter = np.zeros(
            (number_of_feature, number_of_feature))
        for i in range(dataset.number_of_diffrent_label):
            label = dataset.diffrent_label[i]
            class_label = dataset.sample[(dataset.label == label).flatten()]
            size_of_class = dataset.count_diffrent_label[i]
            variance = (class_label-mean_class_feature[i]).T
            within_class_scatter += variance @ variance.T

        w_inv = np.linalg.inv(
            within_class_scatter+0.001*np.eye(number_of_feature))
        w_inv_b = w_inv @ between_class_scatter
        # # It's decreasing sorted by eigenvalues
        # eigenvectors, eigenvalues, vh = np.linalg.svd(w_inv_b)
        # # Using transpose to each row became eigenvalues
        # eigenvectors = eigenvectors.T
        eigenvalues2, eigenvectors2 = np.linalg.eigh(w_inv_b)
        dec_sorted_index = np.flip(np.argsort(eigenvalues2))
        eigenvalues2 = eigenvalues2[dec_sorted_index]
        eigenvectors2 = eigenvectors2.T[dec_sorted_index]
        print('pause')

        self.dataset = dataset
        self.mean_class_feature = mean_class_feature
        self.mean_overall_feature = mean_overall_feature
        self.between_class_scatter = between_class_scatter
        self.within_class_scatter = within_class_scatter
        self.eigenvectors = eigenvectors2
        self.eigenvalues = eigenvalues2
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
        transformed_sample = self.dataset.sample @ eigenvector_subset.T

        self.transformed_sample = transformed_sample

    def reconstruction(self):
        eigenvector_subset = self.eigenvectors[:self.number_of_component]
        reconstructed_sample = self.transformed_sample @ eigenvector_subset
        self.reconstructed_sample = reconstructed_sample

    def visualization(self, images_vector: np.ndarray, title: str = None,
                      random: bool = True):
        images_vector = deepcopy(images_vector)
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
fisher_lda = FisherLDA()
fisher_lda.fit(dataset)
diffrent_k = [1, 6, 29]
fisher_lda.visualization(
    fisher_lda.dataset.sample, 'Fisher_LDA-Images before transformation')
fisher_lda.visualization(
    fisher_lda.eigenvectors, 'Fisher_LDA-Eigenfaces', False)
for k in diffrent_k:
    fisher_lda.transformation(k)
    fisher_lda.reconstruction()
    fisher_lda.visualization(
        fisher_lda.reconstructed_sample,
        f'Fisher_LDA-Images after reconstruction with k = {k}')
print('stop')
