# <center>Homework #4 on Satatistical Patern Recognition</center>

Name: Hesam Mousavi(9931155)

Group: #15

## Module

- ### dataset.py

  - In this module I create class named dataset than contain all variable I need for having dataset (image size, sample, label, label name(name of labels), number of features, and number of samples)

    ``` python
    class Dataset():
        def __init__(self):
            self.image_size = None
            self.sample = None
            self.label = None
            self.label_name = None
            self.number_of_feature = None
            self.number_of_sample = None
    ```

    And have a method in this class to read images from floder and save it in our class(self)

    ``` python
        def read_dataset(self, folder: str, img_size: int,
                     visualize: bool = False):

        images = []
        name_feeling = []

        directory = os.path.abspath('')
        folder_path = os.path.join(directory, folder)
        for img_name in os.listdir(folder_path):
            name_feeling.append(img_name.split('.')[0:-1])
            img_path = os.path.join(folder_path, img_name)
            img_mat = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_mat is not None:
                resized_img_mat = cv2.resize(img_mat, (img_size, img_size))
                images.append(resized_img_mat)

        images = np.array(images)
        name_feeling = np.array(name_feeling).reshape((-1, 3))
        number_of_image = name_feeling.shape[0]

        if visualize is True:
            for rnd in np.random.randint(number_of_image, size=5):
                plt.imshow(images[rnd],  cmap="gray")
                plt.title((
                    f'name: {name_feeling[rnd][0]}, ',
                    f'feeling: {name_feeling[rnd][1]}'))
                plt.show()

        vector_images = images.reshape(images.shape[0], -1)

        self.image_size = img_size
        self.sample = vector_images
        self.label = name_feeling
        self.label_name = np.array(['name', 'feeling'])
        self.number_of_feature = vector_images.shape[1]
        self.number_of_sample = vector_images.shape[0]
    ```

- asd

## Read images

Our dataset here is images of 170 faces dimentions 256x256 and the goal here is to read this images in gray scale mode and resize them to dimentions 64x64

File name here is like XX.YY.000.tiff which XX is short form of person name and YY is short form of his/her feeling

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>
