import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('skin_disease_model.keras')

class_names = [
    "Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", "Atopic Dermatitis", "Bullous Disease", "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema", "Exanthems and Drug Eruptions", "Hair loss Alopecia and other Hair Diseases", "Herpes HPV and other STDs", "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles", "Nail Fungus and other Nail Disease", "Poison Ivy and other Contact Dermatitis", "Psoriasis Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites", "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other ungal Infections", "Hyperpigmentation",
    "Urticaria Hives", "Vascular Tumors", "Vasculitis", "Warts Molluscum and other Viral Infections"
]

def preprocess_image(image_path, target_size=(128, 128)):
    img = image.load_img(image_path, target_size=target_size)  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    return img_array

def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)  
    predicted_class = np.argmax(prediction, axis=-1)  
    predicted_label = class_names[predicted_class[0]]  
    confidence = np.max(prediction) * 100  
    return predicted_label, confidence

image_path = "psoriasis2.jpeg"
predicted_disease, confidence = predict_disease(image_path)

print(f"Predicted Disease: {predicted_disease}")
print(f"Confidence: {confidence:.2f}%")