import ollama
import base64

# Function to encode the image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to get LLaVA response for a given image path
def describe_image(image_path):
    # Encode the image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Run the model with the prompt and image
    response = ollama.chat(
        model="llava",  # Use the LLaVA model
        messages=[
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [image_base64],  # Include the base64-encoded image
            }
        ],
    )

    # Return the response content
    return response["message"]["content"]