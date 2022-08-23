# gcn_clustering
Linkage-based object matching using Graph Convolution Network. One of the advantages of using graph-based approach is that it can make prediction by taking into account of neighbors' information. In this task, we need to predict the relationship between cap, jockey body and saddlecloth bboxes (i.e. which of them belong to the same jockey). Most of the time, these 3 objects are highly overlapped. Given the condition that each jockey can at most have one cap, one body and one saddlecloth, it is possible to group them correctly by looking at neighbors' group (e.g. Given 2 caps placing close to a jockey, if one of them is taken by the neighbor, then we know the other remaining cap should probably be the correct one.). The experimental results show the robustness of this graph-based approach in this challenging object grouping/clustering task.

# Usage
1. Prepare data for triplet model
```
Run prepare_triplet_data.ipynb
```
2. Train triplet model on the triplet data
```
Run triplet_train.ipynb
```
3. Use the best triplet weight to extract graph data
```
Run extract_graph_data.ipynb
```
4. Train the GCN
```
python train.py
```
5. See the prediction result
```
Run Prediction.ipynb
```


# Visualization
In the predictions below, the model looks so robust that it clusters the objects accurately even though the bounding boxes of the objects are highly overlapped. Some of the group colours are close, but if downloading the image and zoom in, the colour difference can be seen.

![gcn1](https://user-images.githubusercontent.com/40629085/169761053-74ed48d6-b38c-4f56-9de8-93af54fe1719.png)
![gcn2](https://user-images.githubusercontent.com/40629085/169761063-302169b8-d606-4c83-abab-17be49da7a99.png)
![gcn3](https://user-images.githubusercontent.com/40629085/175269647-b0c5a369-ffa3-4d14-8f03-3ca607552cc1.png)
![gcn4](https://user-images.githubusercontent.com/40629085/175269659-87227c30-7cf8-4255-b3bb-53cd1ba83ea3.png)
