# Hackaton42

Regle:
* 00_toy_dataset: Seul x compte, si x > 0 alors label 0 sinon label 1
* 01_mnist_cc : Chiffre de gauche = label
* 02_mnist_constant_image: Chiffre de gauche = label
* 03_mnist_constant_image_random_row : On trouve le chiffre de reference et on l'enleve, l'autre chiffre est le label
* 04_mnist_uniform_color_random_row: Le chiffre donne le label peut importe la couleur de fond
* 05_mnist_uniform_color_low_mix_rate: Le chiffre donne le label ==> mettre toutes les images sur le meme fond pour eviter les biais
* 06_mnist_sum: skip
* 07_mnist_sum_bis: delete
* 08_mnist_sum_noise_level: Le chiffre donne le label. Il faut supprimer le bruit
