# GouParichaya: Know Your Desi Cow

Explainable recognition of 50 Indian cattle breeds with a B-cos ResNet-50 (Böhle et al., TPAMI 2024).
The app gives the breed and confidence, says "Not sure" when confidence is low, shows the B-cos
explanation of which image regions were used, and displays a breed information card
(sources: NDDB / ICAR-NBAGR).

The trained model weights are downloaded at startup from a Hugging Face model repository.
