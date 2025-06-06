# Sushmita: Comments 

Overall implementation looks good! 

Some points that could help you look into why the network overfits:

1. Have a validation set and plot the 3 losses for validation as well to confirm overfitting or to see which losses are overfitting. 
2. Depending on what you learn from above, print decoder outputs as images to see how the reconstructions look.
3. Use min-max normalization for images to bring it to (0,1) range and use sigmoid (instead of tanh) as the final activation function for the decoder. 
4. Within a training epoch, try training the auto encoder (encoder+decoder) first, then descriminator and last encoder using adversarial loss. Your current implementation trains descriminator first and then AE which may not be optimal. 
5. The losses for descriminator and adversarial look correct but I would suggest directly using nn.BCELoss instead of implementing it since it offers more safeguards. 
6. Your code has more feature maps in the encoder and decoder than the original implimentation. Why not keep it the same as in the original? 


