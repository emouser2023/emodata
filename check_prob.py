import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
import numpy as np

class LinearProbe(nn.Module):
    def __init__(self, input_dim_per_modality=512, num_modalities=3, hidden_dim=512, num_classes=7, learning_rate=0.01, epochs=100):
        super(LinearProbe, self).__init__()
        total_input_dim = input_dim_per_modality * num_modalities  # 1536

        self.fusion_layer = nn.Linear(total_input_dim, hidden_dim)
        self.clf = nn.Linear(hidden_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(dtype=torch.float16, device=self.device)

    def forward(self, visual_features, text_features, audio_features):
        visual_features = visual_features.to(self.device, dtype=torch.float16)
        text_features = text_features.to(self.device, dtype=torch.float16)
        audio_features = audio_features.to(self.device, dtype=torch.float16)

        combined_features = torch.cat((visual_features, text_features, audio_features), dim=1)
        hidden = torch.relu(self.fusion_layer(combined_features))
        outputs = self.clf(hidden)
        return outputs

    def fit(self, visual_features, text_features, audio_features, targets):
        # Ensure targets are long (integer) for CrossEntropyLoss
        targets = targets.to(self.device, dtype=torch.long)

        self.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self(visual_features, text_features, audio_features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    def predict(self, visual_features, text_features, audio_features):
        self.eval()
        with torch.no_grad():
            outputs = self(visual_features, text_features, audio_features)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.cpu()

    def predict_proba(self, visual_features, text_features, audio_features):
        self.eval()
        with torch.no_grad():
            outputs = self(visual_features, text_features, audio_features)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().float().numpy()

    def evaluate(self, visual_features, text_features, audio_features, targets):
        # Convert targets to integer type (long) before numpy conversion
        targets = targets.to(dtype=torch.long).cpu().numpy()  # Ensure integer type
        pred_labels = self.predict(visual_features, text_features, audio_features).numpy()
        pred_probs = self.predict_proba(visual_features, text_features, audio_features)

        # Metrics
        accuracy = accuracy_score(targets, pred_labels)
        f1 = f1_score(targets, pred_labels, average='macro')
        map_score = average_precision_score(np.eye(7)[targets], pred_probs, average='macro')
        auc = roc_auc_score(np.eye(7)[targets], pred_probs, average='macro', multi_class='ovr')

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mAP': map_score,
            'AUC': auc
        }

# Example usage with your data
# Training data (assuming float16 for features, but targets should be integer-like)
all_visual_features = torch.randn(10098, 512, dtype=torch.float16).cuda()
all_text_features = torch.randn(10098, 512, dtype=torch.float16).cuda()
all_audio_features = torch.randn(10098, 512, dtype=torch.float16).cuda()
all_targets = torch.randint(0, 7, (10098,), dtype=torch.long).cuda()  # Use long for labels

# Testing data
test_visual_features = torch.randn(1098, 512, dtype=torch.float16).cuda()
test_text_features = torch.randn(1098, 512, dtype=torch.float16).cuda()
test_audio_features = torch.randn(1098, 512, dtype=torch.float16).cuda()
test_targets = torch.randint(0, 7, (1098,), dtype=torch.long).cuda()  # Use long for labels

# Initialize the probe
probe = LinearProbe(
    input_dim_per_modality=512,
    num_modalities=3,
    hidden_dim=512,
    num_classes=7,
    learning_rate=0.01,
    epochs=200
)

# Train on all three modalities
probe.fit(all_visual_features, all_text_features, all_audio_features, all_targets)

# Evaluate on test data
metrics = probe.evaluate(test_visual_features, test_text_features, test_audio_features, test_targets)
print("Multimodal Metrics:", metrics)