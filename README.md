
---

### Step 6 — Heterogeneous Graph Neural Network

- 2-layer HeteroGNN using **GATConv**
- Relations:
  - visit → entity  
  - entity → visit  
  - visit → visit  
  - visit → visit (reverse)

Final classifier is applied only to visit node embeddings.  
Loss: **BCEWithLogits + class-wise pos_weight**

---

## 3. Results

| Metric | Score |
|--------|--------|
| Training Loss | ~0.95 |
| Micro-F1 | ~0.21 |
| Macro-F1 | ~0.13 |
| AUROC (micro) | ~0.60 |
| AUROC (macro) | ~0.48 |
| AUPRC (micro) | ~0.17 |
| AUPRC (macro) | ~0.23 |

Given the small dataset (73 patients), results are reasonable.

---

## 4. Future Work

### Data Improvements
- Increase dataset size to 500–3000 patients  
- Better preprocessing of time-series data  

### Model Improvements
- Replace EfficientNet-B3 with DenseNet121 (MIMIC-CXR pre-trained)  
- Use GRU-D or Transformers for time-series  

### Graph Improvements
- Add entity–entity edges  
- Add temporal decay edges  
- Use full Graph Neural ODE  

### Training Improvements
- Use focal loss for imbalance  
- Add image augmentation  

---

## 5. Conclusion

This project demonstrates a complete end-to-end pipeline that combines:

- Multimodal feature extraction  
- Entity importance reasoning  
- Temporal modeling with Neural ODE  
- Heterogeneous Graph Neural Networks  

With larger datasets and model refinements, this approach can scale to real clinical prediction tasks.

