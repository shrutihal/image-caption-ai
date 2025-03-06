import torch
from PIL import Image
from models.blip_model import load_blip_model

# Load model and processor
processor, model = load_blip_model()

def generate_captions(image, num_captions=5, temperature=0.7, top_k=50, top_p=0.9):
    """
    Generates multiple diverse captions for a given image using the BLIP model.

    This function takes an image as input, processes it using the BLIP image 
    processor, and generates multiple caption variations using random sampling 
    techniques. The sampling process is controlled by hyperparameters such as 
    `temperature`, `top_k`, and `top_p` to balance creativity and accuracy.

    Args:
        image (PIL.Image.Image): The input image for which captions need to be generated.
        num_captions (int, optional): Number of unique captions to generate. Default is 5.
        temperature (float, optional): Controls randomness in generation. Higher values 
            (e.g., 1.0) produce more diverse captions, while lower values (e.g., 0.3) 
            make the output more deterministic. Default is 0.7.
        top_k (int, optional): Limits sampling to the top-k most likely words at each step. 
            Lower values (e.g., 10) make the output more focused, while higher values 
            (e.g., 100) introduce more diversity. Default is 50.
        top_p (float, optional): Nucleus sampling; considers only the top-p cumulative 
            probability mass when selecting words. Values closer to 1.0 allow for more 
            diverse generations. Default is 0.9.

    Returns:
        list: A list of `num_captions` generated captions, each represented as a string.

    Raises:
        ValueError: If `num_captions` is less than 1.
        RuntimeError: If the model fails to generate captions due to CUDA/memory issues.

    """
    if num_captions < 1:
        raise ValueError("num_captions must be at least 1.")

    inputs = processor(images=image, return_tensors="pt").to(model.device)

    captions = []
    for _ in range(num_captions):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,  # Enables randomness in generation
                max_length=50, # characters limitation
                temperature=temperature,  # Controls randomness
                top_k=top_k,  # Limits choices to top K words
                top_p=top_p  # Nucleus sampling
            )
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        captions.append(caption)

    return captions
