# WiFiGEN

> Indoor imaging is a critical task for robotics and internet-of-things. WiFi as an omnipresent signal is a promising candidate for carrying out passive imaging and synchronizing the up-to-date information to all connected devices. This is the first research work to consider WiFi indoor imaging as a multi-modal image generation task that converts the measured WiFi power into a high-resolution indoor image. Our proposed WiFi-GEN network achieves a shape reconstruction accuracy that is 275% of that achieved by physical model-based inversion methods. Additionally, the Fr´echet Inception Distance score has been significantly reduced by 82%. To examine the effectiveness of models for this task, the first large-scale dataset is released containing 80,000 pairs of WiFi signal and imaging target. Our model absorbs challenges for the model-based methods including the non-linearity, ill-posedness and non-certainty into massive parameters of our generative AI network. The network is also designed to best fit measured WiFi signals and the desired
imaging output. For reproducibility, we will release the data and code upon acceptance.
  
## Recent Updates
- [x] Initial code release  
- [x] Release training code

## 代码说明 
- utils/readmat.py # Process MAT files into data pairs suitable for model training.
