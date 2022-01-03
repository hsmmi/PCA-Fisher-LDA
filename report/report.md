# <center>Homework #4 on Satatistical Patern Recognition</center>

Name: Hesam Mousavi(9931155)

Group: #15

## Module

- ### dataset.py

  - In this module I create class named dataset than contain all variable I need for having dataset (image size, sample, label, diffrent lable, number of diffrent lable, label name(name of labels), number of features, and number of samples)

    And have a method in this class to read images from floder and save it in our class(self)

- ### pca.py

  - In this module I create class named PCA than contain all variable I need for having dataset (image size, sample, label, diffrent lable, number of diffrent lable, label name(name of labels), number of features, and number of samples)

## Read images

Our dataset here is images of 170 faces dimentions 256x256 and the goal here is to read this images in gray scale mode and resize them to dimentions 64x64

File name here is like XX.YY0.000.tiff which XX is short form of person name and YY is short form of his/her feeling

## PCA

It's unsupervised problem that wants several purposes like making visualization possible or reducing training time, .. by reducing dimensions

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
