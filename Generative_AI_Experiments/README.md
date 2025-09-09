# Neural Style Transfer using VGG19

This project implements **neural style transfer** to generate images that combine the **content** of a source image with the **style** of a reference image. The project uses the **pre-trained VGG19** network and gradient-based optimization to iteratively update the generated image.

---

## Project Overview

Neural style transfer separates **content** and **style** of images:

- **Content**: The structure and objects in the original image.
- **Style**: Texture, colors, and patterns from a reference image.

This approach allows creating visually appealing hybrid images that maintain the original scene while adopting a new artistic style.

---

## Methodology

1. **Load Pre-trained VGG19**
   - Utilizes ImageNet weights.
   - Extract feature maps from specific layers to compute style and content.

2. **Loss Functions**
   - **Content Loss**: Maintains the structure of the base image.
   - **Style Loss**: Uses Gram matrices to encode and transfer style.
   - **Total Variation Loss**: Encourages spatial smoothness and reduces noise.

3. **Preprocessing**
   - Resize and normalize images.
   - Convert images to tensors compatible with VGG19.

4. **Optimization**
   - Generate an image initialized from the content image.
   - Update the generated image using **SGD** with an **exponential learning rate decay**.
   - Run for **4000 iterations** with intermediate outputs saved every 100 steps.

5. **Dataset**
   - **Content images:** 7 custom images (e.g., family photos, landscapes).
   - **Style references:** 7 artistic style images (e.g., `scream.jpg`, `starry_night.jpg`).

6. **Results**
   - Progressive optimization logs:
     ```
     Iteration 100: loss=2058.96
     Iteration 500: loss=775.63
     Iteration 1000: loss=628.97
     Iteration 2000: loss=551.44
     Iteration 4000: loss=510.70
     ```
   - Final generated images blend content and style effectively.
   - Outputs stored in `results/` and archived in `results.zip`.
   - Style reference images archived in `style_refs.zip`.

---

## Example

**Content Image:** `image1_me_and_sis.jpg`  
**Style Reference:** `scream.jpg`  

The generated image preserves the original content while adopting the textures and colors of the reference style.

---


---

## Tools & Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy
- PIL (Pillow)
- Matplotlib (for visualization)

---

## Key Takeaways

- Demonstrates **deep learning for artistic image generation**.
- Shows how **pre-trained CNNs** can separate content and style.
- Illustrates **iterative optimization** with gradient descent for image synthesis.

## Results
Results are under the Releases tab v1.0.0

