## Dataset
The dataset is constructed based on publicly available data [1]:

- 1,470 users who have reviewed at least 5 products were randomly sampled from the Amazon_Fashion category of the Amazon Reviews dataset, available at [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/) [1].
- These users collectively reviewed 12,404 products, resulting in a total of 15,127 user-product edges.
- Each edge represents a user’s interaction (i.e., review) with a product, forming a bipartite graph between users and products.
- Product features are provided in the dataset, which are further converted to one-hot product features, while user features were randomly initialized.
- A directed edge is created from a product to a persona if the product has been purchased at least twice by users belonging to that persona.

### Reference

[1] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley.  
*Bridging Language and Items for Retrieval and Recommendation*. arXiv preprint arXiv:2403.03952, 2024.

