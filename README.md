# 🔗 Community Detection in Hypergraphs

## 📌 Overview
This project focuses on **community detection in hypergraph networks** using three different approaches:
- IRMM (Iteratively Reweighted Modularity Maximization)
- FMHMO (Fuzzy Membership-based Hypergraph Modularity Optimization)
- HGNN (Hypergraph Neural Network)

The goal is to compare these methods and analyze their performance in detecting meaningful communities in complex networks.

---

## 🎯 Objectives
- Understand community detection in hypergraphs  
- Implement IRMM, FMHMO, and HGNN methods  
- Compare clustering quality and performance  
- Analyze which method gives better results  

---

## 📂 Dataset
We use a **Gene-Disease Hypergraph Dataset**, which represents relationships between genes and diseases.

- Nodes: 9,262  
- Edges: 886  
- Hyperedges: 2,242  
- Type: Biological Network  

Each node represents a gene or disease, and connections represent their relationships.

---

## ⚙️ Methods Used

### 🔹 IRMM
- Converts hypergraph into weighted graph  
- Uses Louvain algorithm for clustering  
- Iteratively updates weights  

### 🔹 FMHMO
- Uses fuzzy memberships  
- Allows overlapping communities  
- Produces more accurate clustering  

### 🔹 HGNN
- Deep learning-based approach  
- Learns node embeddings  
- Uses clustering (K-Means) to detect communities  

---

## 🛠️ Tech Stack
- Python  
- NetworkX  
- NumPy & Pandas  
- Scikit-learn  
- PyTorch / PyTorch Geometric  
- Matplotlib  

---

## 🚀 Workflow
1. Load and preprocess dataset  
2. Construct hypergraph  
3. Apply IRMM, FMHMO, and HGNN  
4. Generate communities  
5. Compare results  

---

## 📊 Results
- IRMM → Basic clustering  
- FMHMO → Best clarity and accuracy  
- HGNN → Powerful but complex  

---

## 📈 Evaluation Criteria
- Clustering Quality  
- Community Clarity  
- Performance  

---

## 🔮 Future Scope
- Apply on larger datasets  
- Improve model efficiency  
- Use advanced deep learning models  

---

## 👨‍💻 Author
**Tanuj Garia**

---

## ⭐ Note
This project demonstrates how advanced techniques like fuzzy logic and deep learning improve community detection in complex hypergraph networks.
