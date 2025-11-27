# Multimodal Temporal Graph Learning for Disease Prediction

This project builds a multimodal and temporal patient representation to predict 25 clinical outcomes. We integrate **clinical text**, **chest X-ray images**, and **time-series vitals** into a unified **heterogeneous graph**, refine visit representations using **entity-importance attention**, model temporal progression using a **Neural ODE**, and finally train a **heterogeneous GNN** for multilabel prediction. 

---

## 🚀 Pipeline Overview

### **1. Clinical Entity Extraction**
- Uses `samrawal/bert-base-uncased-clinical-ner` to extract diseases/conditions from Past Medical History (PMH) text.  
- Entities are deduplicated and embedded using BERT CLS vectors (768-dim).

### **2. Multimodal Visit Embeddings**
Each hospital visit is encoded using:
- **Text:** BERT embedding (768-dim)  
- **Image:** EfficientNet-B3 embedding (1536-dim)  
- **Time-series:** numeric statistics + categorical embeddings  
Combined visit embedding:  


### **3. Building the Visit–Entity Graph**
For each patient, a heterogeneous graph is constructed with:
- Visit nodes  
- Entity nodes  
- Edges: visit→entity, entity→visit, visit↔visit (temporal order)  
Saved as PyG `HeteroData`. 

### **4. Entity-Importance Attention Update**
Each connected clinical entity receives an importance score based on:
- Similarity to visit text  
- Similarity to outcome labels  

Visit text embedding is updated as a weighted sum of entity embeddings.  
Graphs stored in `input_graphs_attn/`. 

### **5. Neural ODE Temporal Modeling**
A Neural ODE models continuous-time progression between visits:
Updated graphs saved in `input_graphs_neural_ode/`. 

### **6. Heterogeneous Graph Neural Network**
A 2-layer HeteroGNN (GAT-based) is trained using relations:
- visit → entity  
- entity → visit  
- visit → visit  
- visit → visit (reverse)  

Loss: **BCEWithLogits** with class-wise positive weights.  
Training: 100 epochs, patience 20. 

---

## 📊 Results
Final performance (micro/macro):  
- **F1:** 0.21 / 0.13  
- **AUROC:** 0.60 / 0.48  
- **AUPRC:** 0.17 / 0.23  


Given only **73 patients**, the model performs reasonably and shows potential.

---

## 🔧 Future Improvements
- Increase dataset size (500–3000 patients)  
- Replace EfficientNet-B3 with medical models (e.g., DenseNet121 MIMIC-CXR)  
- Use GRU-D or Transformer-based time-series embeddings  
- Add entity–entity edges and temporal decay  
- Use full Graph Neural ODE  
- Apply focal loss + better augmentation  


---

## ✅ Conclusion
This project presents a complete multimodal temporal graph learning pipeline, combining NER, multimodal embeddings, entity-aware refinement, temporal ODE modeling, and heterogeneous GNNs for disease prediction. With more data and architectural enhancements, this approach can scale to real clinical prediction tasks. 
