# Hackaton42
This was a hackathon of 2.5 days organized by Effisciences in the 42 School in Paris. 

The task was to obtain a sufficent accuracy on a validation set while the training set distribution was different in terms of correlation : in the training set, two or more features always were highly correlated relatively to the eval test. This is a very common in applied Machine Learning where the practical context is different than the training settings, e.g . a model classifying images as "cows" or "cammels" could rather learn to classify images as "grass fields landscape" or "desert landscape".

Much more details are given here : https://github.com/EffiSciencesResearch/hackathon42

We scored 5th out of 50 teams. Our method was to find the actual evaluation rule and preprocess the training/eval data so that the correlation difference disappear without losing performance/

### Rules:
* 00_toy_dataset: Seul x compte, si x > 0 alors label 0 sinon label 1
* 01_mnist_cc : Chiffre de gauche = label
* 02_mnist_constant_image: Chiffre de gauche = label
* 03_mnist_constant_image_random_row : On trouve le chiffre de reference et on l'enleve, l'autre chiffre est le label
* 04_mnist_uniform_color_random_row: Le chiffre donne le label peut importe la couleur de fond
* 05_mnist_uniform_color_low_mix_rate: Le chiffre donne le label ==> mettre toutes les images sur le meme fond pour eviter les biais
* 06_mnist_sum: skip
* 07_mnist_sum_bis: delete
* 08_mnist_sum_noise_level: Le chiffre donne le label. Il faut supprimer le bruit
