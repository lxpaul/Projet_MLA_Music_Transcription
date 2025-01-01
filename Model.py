
# Définir la fonction de construction du modèle NMP
def build_nmp_model(input_shape):
    """
    Construit le modèle NMP selon la description donnée.
    Args:
        input_shape: Tuple décrivant la forme de l'entrée (hauteur, largeur, canaux).
    Returns:
        Un modèle Keras compilé.
    """
    # Entrée audio transformée (CQT)
    inputs = Input(shape=input_shape, name="CQT_Input")

    # Bloc 1: Extraction de caractéristiques initiales
    x = Conv2D(32, (5, 5), strides=(1, 3), padding="same", name="Conv2D_Block1_1")(inputs)
    x = BatchNormalization(name="BatchNorm_Block1_1")(x)
    x = ReLU(name="ReLU_Block1_1")(x)

    # Bloc 2: Extraction des activations multipitch (Yp)
    yp = Conv2D(16, (5, 5), padding="same", name="Conv2D_Block2_1")(inputs)
    yp = BatchNormalization(name="BatchNorm_Block2_1")(yp)
    yp = ReLU(name="ReLU_Block2_1")(yp)

    yp = Conv2D(8, (3, 39), padding="same", name="Conv2D_Block2_2")(yp)
    yp = BatchNormalization(name="BatchNorm_Block2_2")(yp)
    yp = ReLU(name="ReLU_Block2_2")(yp)

    yp = Conv2D(1, (5, 5), activation="sigmoid", padding="same", name="Yp")(yp)

    # Bloc 3: Extraction des activations de notes (Yn)
    yn = Conv2D(32, (7, 7), strides=(1, 3), padding="same", name="Conv2D_Block3_1")(yp)
    yn = ReLU(name="ReLU_Block3_1")(yn)

    yn = Conv2D(1, (7, 3), activation="sigmoid", padding="same", name="Yn")(yn)

    # Bloc 4: Détection des débuts de notes (Yo)
    yo = Concatenate(name="Concat_Yo")([x, yn])
    yo = Conv2D(1, (3, 3),activation="sigmoid", padding="same", name="Yo")(yo)
    

    # Définir le modèle avec les trois sorties
    model = Model(inputs=inputs, outputs=[yo, yn, yp], name="NMP_Model")

 
    
       # Créer une instance de la perte pondérée pour Yo
    weighted_loss = WeightedBinaryCrossEntropy(positive_weight=0.95, negative_weight=0.05)
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "Yo": weighted_loss,
            "Yn": "binary_crossentropy",
            "Yp": "binary_crossentropy",
        },
        loss_weights={"Yo": 1.0, "Yn": 1.0, "Yp": 1.0},
        metrics={"Yo": "accuracy", "Yn": "accuracy", "Yp": "accuracy"}
    )


    return model


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, positive_weight=0.95, negative_weight=0.05, name="weighted_binary_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def call(self, y_true, y_pred):
        """
        Appliquer une binary cross-entropy pondérée.
        """
        # Supprimer la dimension supplémentaire de y_pred si nécessaire
        y_pred = tf.squeeze(y_pred, axis=-1)

        # Calcul manuel de la binary cross-entropy (éviter toute réduction prématurée)
        bce = -(
            y_true * tf.math.log(y_pred + 1e-7) +
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        )

        # Calcul des poids
        weights = y_true * self.positive_weight + (1 - y_true) * self.negative_weight

        # Appliquer les poids
        weighted_bce = weights * bce

        # Calcul final : moyenne sur toutes les dimensions
        return tf.reduce_mean(weighted_bce)

    def get_config(self):
        """
        Configuration pour la sérialisation.
        """
        config = super().get_config()
        config.update({
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruction de l'objet à partir de la configuration.
        """
        return cls(**config)


# Exemple d'initialisation du modèle
input_shape = (181, 264, 1)  # Exemple de taille d'entrée (CQT avec un canal)
model = build_nmp_model(input_shape)

# Afficher le résumé du modèle
model.summary()
print("Model outputs:", model.output_names)


