import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_blip_model():
    """
    Loads the BLIP (Bootstrapped Language-Image Pretraining) model and processor.

    The BLIP model is a state-of-the-art vision-language model used for 
    image captioning and other multimodal tasks. This function loads the 
    pre-trained BLIP model from Hugging Face along with its associated 
    image processor.

    Returns:
        tuple: A tuple containing:
            - `BlipProcessor`: The processor used to preprocess images before 
              feeding them into the BLIP model.
            - `BlipForConditionalGeneration`: The BLIP model itself, capable of 
              generating captions based on input images.

    Raises:
        OSError: If the model cannot be loaded due to internet connection issues 
        or invalid model name.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    return processor, model
