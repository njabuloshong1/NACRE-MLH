"""Architecture of the NACRE resolver: layer stack, tensor shapes and gradient flow.\n\nKept in step with nacre/nacre_ml.py, which builds the same stack:\n    Linear(d,256) -> BatchNorm -> ReLU -> Dropout(0.3)\n    Linear(256,128) -> BatchNorm -> ReLU -> Dropout(0.3)\n    Linear(128,C)\n\nd and C depend on the reference, so the diagram names them rather than fixing numbers.\n\n    python docs/mlp_diagram.py  ->  docs/resolver_architecture.png\n"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
fig,ax=plt.subplots(figsize=(19,9)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

# blocks: (label, sublabel, facecolor)
blocks=[
 ("Input  x","[ E | P | M ]\nd = 52 + C\n(C = reference cell types)","#E8ECF1"),
 ("Linear\nd → 256","W1 (256×d), b1\n(learnable)","#F3B76B"),
 ("BatchNorm → ReLU\n→ Dropout(0.3)","stabilize • nonlinearity\n• regularize","#AEC7E8"),
 ("Linear\n256 → 128","W2 (128×256), b2\n(learnable)","#F3B76B"),
 ("BatchNorm → ReLU\n→ Dropout(0.3)","","#AEC7E8"),
 ("Linear\n128 → C","W3 (C×128), b3\n(learnable)","#F3B76B"),
 ("Softmax","logits → probabilities","#C4A7E7"),
 ("Weighted\nCross-Entropy Loss","−w_t · log p_t","#E7897B"),
]
shapes=["x: (d)","(256)","(256)","(128)","(128)","(C) logits","(C) probs","scalar L"]
n=len(blocks); cx=np.linspace(0.085,0.915,n); w=0.092; h=0.17; yc=0.63
xr=[]; xl=[]
for i,(lab,sub,fc) in enumerate(blocks):
    box=FancyBboxPatch((cx[i]-w/2,yc-h/2),w,h,boxstyle="round,pad=0.006,rounding_size=0.012",
                       fc=fc,ec="#333",lw=1.4); ax.add_patch(box)
    ax.text(cx[i],yc+0.028,lab,ha="center",va="center",fontsize=9.2,fontweight="bold")
    if sub: ax.text(cx[i],yc-0.045,sub,ha="center",va="center",fontsize=6.6,color="#333")
    xl.append(cx[i]-w/2); xr.append(cx[i]+w/2)
# forward arrows (blue) + shape labels
for i in range(n-1):
    ax.add_patch(FancyArrowPatch((xr[i],yc),(xl[i+1],yc),arrowstyle="-|>",mutation_scale=18,
                 lw=2.2,color="#2E6FB7"));
    ax.text((xr[i]+xl[i+1])/2,yc+0.105,shapes[i+1],ha="center",va="bottom",fontsize=7,color="#2E6FB7",fontweight="bold")
ax.text(0.5,0.895,"FORWARD PASS  ▶",ha="center",fontsize=12,color="#2E6FB7",fontweight="bold")

# backward arrows (red dashed) underneath, from Loss back through the learnable layers
yb=0.34
for i in range(n-1,0,-1):
    a=FancyArrowPatch((cx[i],yc-h/2-0.01),(cx[i-1],yb),arrowstyle="-|>",mutation_scale=13,lw=1.3,
                      color="#C0392B",ls="--",connectionstyle="arc3,rad=0.28"); ax.add_patch(a)
ax.text(0.5,0.245,"◀  BACKWARD PASS (backpropagation): compute ∂L/∂θ for every weight by the chain rule",
        ha="center",fontsize=11,color="#C0392B",fontweight="bold")
ax.text(0.5,0.205,"Adam optimizer updates W1,b1, W2,b2, W3,b3 and BatchNorm params  (lr = 1e-3, weight decay = 1e-4)",
        ha="center",fontsize=9,color="#C0392B")

# input composition note
ax.text(cx[0],0.40,"E = expression SVD (50), z-scored\nP = spatial (x,y) × 0.3\nM = neighborhood cell-type\ncomposition over 15-NN  (C dims)",
        ha="center",va="top",fontsize=7.4,color="#333",
        bbox=dict(boxstyle="round,pad=0.4",fc="#F6F8FA",ec="#AAB"))
# output / inference note
ax.text(cx[6],0.40,"At inference (split cells):\nsoftmax → probabilities →\nrestrict to the 4 tools' candidate\nlabels → argmax → FINAL label",
        ha="center",va="top",fontsize=7.4,color="#1a6b3a",
        bbox=dict(boxstyle="round,pad=0.4",fc="#EAF6EE",ec="#8CC5A2"))

# training-data banner
ax.text(0.5,0.115,"Trained (supervised) on the CONFIDENT cells (lock 4/4 + majority 3/1) — target = consensus cell-type label · "
        "90/10 train–val · early stopping (patience 20)\nThe contested SPLIT cells are never in training; they are what the trained network predicts.",
        ha="center",va="center",fontsize=8.6,color="#222",
        bbox=dict(boxstyle="round,pad=0.5",fc="#FFF9E8",ec="#E3C97A"))
# legend for box colors
leg=[("#F3B76B","Linear (learnable weights)"),("#AEC7E8","BatchNorm/ReLU/Dropout"),
     ("#C4A7E7","Softmax"),("#E7897B","Loss"),("#E8ECF1","Input feature vector")]
for i,(c,t) in enumerate(leg):
    ax.add_patch(FancyBboxPatch((0.055+i*0.19,0.035),0.02,0.02,boxstyle="round,pad=0.002",fc=c,ec="#333"))
    ax.text(0.08+i*0.19,0.045,t,fontsize=7.3,va="center")
fig.suptitle("NACRE-ML refiner — MLP forward & backward pass  (input: expression+space+neighborhood → output: cell-type probabilities)",
             fontsize=13.5,fontweight="bold",y=0.98)
fig.savefig("resolver_architecture.png",dpi=170,bbox_inches="tight")
print("saved -> resolver_architecture.png")
