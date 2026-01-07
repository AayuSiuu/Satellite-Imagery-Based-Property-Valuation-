import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD TEST TABULAR DATA
# -----------------------------
test_df = pd.read_excel("data/raw/test2.xlsx")

features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "sqft_living15", "sqft_lot15",
    "condition", "grade", "view", "waterfront"
]

X_tab = test_df[features]

# -----------------------------
# SCALE TABULAR FEATURES
# -----------------------------
scaler = joblib.load("scaler.pkl")
X_tab_scaled = scaler.transform(X_tab)

# -----------------------------
# LOAD IMAGE EMBEDDINGS
# -----------------------------
X_img = np.load("data/processed/image_embeddings.npy")

# ALIGN TABULAR TO IMAGE COUNT
num_img = X_img.shape[0]
X_tab_scaled = X_tab_scaled[:num_img]

print("Aligned tabular shape:", X_tab_scaled.shape)
print("Image embeddings shape:", X_img.shape)

# -----------------------------
# MULTIMODAL FUSION
# -----------------------------
X_fused = np.concatenate([X_tab_scaled, X_img], axis=1)
print("Inference fused shape:", X_fused.shape)

# -----------------------------
# DEFINE MODEL (MUST MATCH TRAINING)
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, tab_dim, img_dim):
        super().__init__()

        self.tabular_branch = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.image_branch = nn.Sequential(
            nn.Linear(img_dim, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 448, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_tab, x_img):
        t = self.tabular_branch(x_tab)
        i = self.image_branch(x_img)
        x = torch.cat([t, i], dim=1)
        return self.fusion(x)

# -----------------------------
# LOAD PYTORCH MODEL
# -----------------------------
checkpoint = torch.load("fusion_model.pth", map_location=device)

tab_dim = checkpoint["tabular_dim"]
img_dim = checkpoint["image_dim"]

model = FusionModel(tab_dim, img_dim).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# INFERENCE
# -----------------------------
with torch.no_grad():
    preds_log = model(
        torch.tensor(X_tab_scaled, dtype=torch.float32).to(device),
        torch.tensor(X_img, dtype=torch.float32).to(device)
    ).cpu().numpy().squeeze()


# LOG → REAL PRICE
preds = np.expm1(preds_log)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
pd.DataFrame({
    "id": test_df.index[:len(preds)],
    "predicted_price": preds
}).to_csv("predictions.csv", index=False)

print("✅ Multimodal predictions.csv generated using fusion_model.pth")
