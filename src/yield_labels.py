import numpy as np

yield_loss_map = {
    "Apple___Apple_scab":                                    30.0,
    "Apple___Black_rot":                                     45.0,
    "Apple___Cedar_apple_rust":                              35.0,
    "Apple___healthy":                                        0.0,
    "Blueberry___healthy":                                    0.0,
    "Cherry_(including_sour)___Powdery_mildew":              25.0,
    "Cherry_(including_sour)___healthy":                      0.0,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":    40.0,
    "Corn_(maize)___Common_rust_":                           25.0,
    "Corn_(maize)___Northern_Leaf_Blight":                   45.0,
    "Corn_(maize)___healthy":                                 0.0,
    "Grape___Black_rot":                                     50.0,
    "Grape___Esca_(Black_Measles)":                          60.0,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":            40.0,
    "Grape___healthy":                                        0.0,
    "Orange___Haunglongbing_(Citrus_greening)":              70.0,
    "Peach___Bacterial_spot":                                35.0,
    "Peach___healthy":                                        0.0,
    "Pepper,_bell___Bacterial_spot":                         30.0,
    "Pepper,_bell___healthy":                                 0.0,
    "Potato___Early_blight":                                 30.0,
    "Potato___Late_blight":                                  55.0,
    "Potato___healthy":                                       0.0,
    "Raspberry___healthy":                                    0.0,
    "Soybean___healthy":                                      0.0,
    "Squash___Powdery_mildew":                               25.0,
    "Strawberry___Leaf_scorch":                              30.0,
    "Strawberry___healthy":                                   0.0,
    "Tomato___Bacterial_spot":                               35.0,
    "Tomato___Early_blight":                                 25.0,
    "Tomato___Late_blight":                                  65.0,
    "Tomato___Leaf_Mold":                                    17.0,
    "Tomato___Septoria_leaf_spot":                           32.0,
    "Tomato___Spider_mites Two-spotted_spider_mite":         20.0,
    "Tomato___Target_Spot":                                  28.0,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":                55.0,
    "Tomato___Tomato_mosaic_virus":                          40.0,
    "Tomato___healthy":                                       0.0,
}

def get_yield_loss(disease_name):
    base = yield_loss_map.get(disease_name, 5.0)
    rng  = np.random.default_rng(abs(hash(disease_name)) % (2**32))
    noise = rng.uniform(-3.0, 3.0)
    return round(base + noise, 1)