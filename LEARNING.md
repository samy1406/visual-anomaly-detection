# 📚 LEARNING.md — Visual Anomaly Detection

> Personal notes from building this project. Not polished — raw understanding.
> These are the concepts interviewers will probe. Own every section.

---

## 1. Why Anomaly Detection is Hard (and Different)

**Standard CV** = supervised, balanced classes, labels for everything.  
**Anomaly detection** = unsupervised, heavily imbalanced, defects are rare & varied.

**Think about this:**  
- Why can't you just train a binary classifier (normal vs defect)?  
- What's the real problem with collecting defect labels in manufacturing?

**Hint:** Defects are *open-set* — new defect types appear at deployment. A classifier trained on known defects will miss novel ones.

---

## 2. MVTec AD Dataset — Understand It Before You Code

- 15 categories: bottles, cables, carpets, etc.
- Training set: **normal images only** (this is intentional!)
- Test set: normal + 10+ defect types per category
- Ground truth: pixel-level masks for defects

**Explore before modelling:**
- How many images per category? Per split?
- What's the image resolution? Does it matter?
- Which categories are "texture" vs "object"? Why does this distinction matter for your model?

**Do this manually:** Pick the `bottle` category. Visually look at 5 normal and 5 defective images. What does "broken_large" look like vs "contamination"?

---

## 3. Image Preprocessing Pipeline

```python
# Standard for ImageNet-pretrained backbone
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Question to answer yourself:**
- Why ImageNet mean/std normalization? What does it do mathematically?
- Why CenterCrop vs RandomCrop at test time?
- PatchCore uses 224×224 or 256×256 input? Try both — does it affect AUROC?

---

## 4. Approach A — AutoEncoder (Reconstruction-Based)

### Core Idea
Train encoder-decoder only on normal images → learns to reconstruct normal well.  
At test: compute pixel-wise reconstruction error → high error = anomaly.

### Architecture Hint
```
Input (3×224×224)
→ Conv2d (3→32, k=3, stride=2) → ReLU → BN   [112×112]
→ Conv2d (32→64, k=3, stride=2) → ReLU → BN  [56×56]
→ Conv2d (64→128, k=3, stride=2) → ReLU → BN [28×28]
→ [BOTTLENECK — latent representation]
→ ConvTranspose2d upsample back to 3×224×224
```

**You must figure out:**
- How many layers? What bottleneck size?
- Use MSE loss or L1 loss? What's the difference for textures?
- How do you get a per-pixel anomaly map from reconstruction?
- Why does AE struggle with textures? (carpet, tile categories)

**Hint for the texture problem:** AEs learn to reconstruct texture *patterns*, not specific arrangements. So even anomalous textures get reconstructed "well enough."

### Training Loop (CPU safe — use Codespace)
- Batch size: 8–16
- Epochs: 50–100
- Adam optimizer, lr=1e-3, reduce on plateau
- Save best model by val reconstruction loss

---

## 5. Approach B — PatchCore (Memory Bank)

### Core Idea
PatchCore doesn't train a new model. It uses a **pretrained backbone** (ResNet/WideResNet) frozen and extracts **patch-level features** from intermediate layers.

These features form a **memory bank** of "normal patch embeddings."  
At test: extract patches → compare against memory bank via k-NN → max distance = anomaly score.

### Step-by-Step Flow

```
1. Load pretrained ResNet18 (frozen — no gradient)
2. Register hooks on layer2 + layer3 to capture feature maps
3. For each training image:
   - Forward pass → get feature maps [B, C, H, W]
   - Spatially pool to patch features [B, C, p, p]
   - Flatten to patch embeddings [B*p*p, C]
4. Subsample patches (coreset subsampling — avoid redundancy)
5. Build FAISS index from all patch embeddings
6. At test:
   - Extract patches same way
   - For each patch, find k nearest neighbors in memory bank
   - Anomaly score = mean of k-NN distances
   - Reshape scores to spatial grid → heatmap
7. Image-level score = max patch score
```

**Critical concepts to understand yourself:**

**Why intermediate layers (layer2 + layer3)?**  
- Hint: Think about what early vs late layers encode in ResNets.
- Early = local textures/edges. Late = semantic/object-level. Anomalies are spatial and local — which level makes more sense?

**What is coreset subsampling?**  
- The memory bank can get huge (thousands of images × H×W patches).
- Coreset = greedy selection of N points that best represent the full set.
- Why not just random subsample? Think about coverage.

**Why FAISS?**  
- k-NN on CPU over 50,000+ vectors is slow.
- FAISS = Facebook AI Similarity Search — approximate nearest neighbor.
- Use `faiss.IndexFlatL2` to start (exact), then try `IndexIVFFlat` (approximate).

**Question:** PatchCore has no training. So what's the "learning"?

---

## 6. Evaluation — AUROC

### Why AUROC?
- No threshold needed
- Threshold-free: "at all possible thresholds, how well does your score separate normal from anomaly?"
- AUROC = 1.0 → perfect separation. 0.5 → random.

### Compute It

```python
from sklearn.metrics import roc_auc_score

# y_true: 0=normal, 1=anomaly (per image)
# y_score: your anomaly score per image (higher = more anomalous)
auroc = roc_auc_score(y_true, y_score)
```

**Think about:**
- What if you have 100 normal and 5 anomaly test images? Does AUROC handle this? What doesn't it handle?
- How would you pick an operating threshold for a real deployment?
- What's the difference between image-level AUROC and pixel-level AUROC?

---

## 7. Anomaly Heatmap Visualization

### For AutoEncoder
```python
# reconstruction_error per pixel
anomaly_map = torch.abs(input_img - reconstructed_img).mean(dim=0)
```

### For PatchCore
```python
# anomaly_scores shape: [H_patches, W_patches]
# Upsample to image resolution
anomaly_map = F.interpolate(scores.unsqueeze(0).unsqueeze(0), 
                            size=(224, 224), 
                            mode='bilinear').squeeze()
```

**Overlay on original:**
```python
import cv2
heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
```

---

## 8. Comparison: PatchCore vs AutoEncoder

| Property | PatchCore | AutoEncoder |
|---|---|---|
| Training needed | ❌ No (feature extraction only) | ✅ Yes (50+ epochs) |
| GPU required for training | ❌ No | ✅ Yes |
| Inference speed | Slower (FAISS k-NN) | Fast (single forward pass) |
| Memory usage | High (memory bank) | Low |
| Texture performance | ⚠️ Moderate | ❌ Struggles |
| Object performance | ✅ Excellent | ✅ Good |
| Explainability | ✅ Spatial heatmap | ✅ Reconstruction diff |
| SOTA AUROC on MVTec | ~99% | ~85% |

**Why does PatchCore dominate?**  
Hint: It leverages features trained on 1.2M ImageNet images. AE starts from scratch on ~200 normal images.

---

## 9. Common Failure Cases (Interview Gold)

- **AE generates sharp anomalies:** If the AE has too much capacity, it can reconstruct anomalies too. Solution: restrict bottleneck.
- **PatchCore memory bank too large:** Slows inference, eats RAM. Solution: aggressive coreset subsampling.
- **Threshold selection:** In deployment, you need a threshold. How? Use validation normal images → set threshold at 95th percentile of scores.
- **Domain shift:** Model trained on bottle will fail on cable. Always category-specific.

---

## 10. Interview Questions You Must Answer Cold

1. **"Why unsupervised for anomaly detection?"**  
   → Defects are rare, varied, often unknown at training time. Supervised requires labeled defect data which is expensive and incomplete.

2. **"How does PatchCore work in 60 seconds?"**  
   → Frozen pretrained backbone extracts patch embeddings from normal images → FAISS memory bank → test patches compared via k-NN → high distance = anomaly.

3. **"Why not use a GAN for this?"**  
   → GANs can work (AnoGAN) but training is unstable and slow inference. PatchCore is simpler, faster, and more accurate.

4. **"What does AUROC not tell you?"**  
   → It doesn't tell you the operating threshold, precision at a given recall, or how the model performs at a specific deployment operating point.

5. **"What's the bottleneck in your PatchCore pipeline?"**  
   → Memory bank size. On large datasets or fine-grained patches, k-NN search becomes slow. FAISS approximate search and coreset subsampling address this.

6. **"Your AUROC is 0.92 but the client says too many false positives. What do you do?"**  
   → Lower the threshold (increase specificity at cost of sensitivity). Or look at precision-recall curve, not just AUROC. Or add a second-stage classifier on flagged images.

---

## 11. Project-Specific Learnings (Fill This In Yourself)

> After you complete the project, write your own answers here.

- What AUROC did you actually achieve on bottle? on carpet?
- Which was the hardest category and why?
- Did you try WideResNet50 backbone? How much did it help?
- What was the most surprising failure mode you found?
- How long did FAISS indexing take on Colab vs Codespace?
- Did you visualize wrong predictions? What did they look like?

---

*Last updated: May 2026*
