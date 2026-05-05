from graphviz import Digraph
import os

# =========================================================
# Generate Flowchart for 4-Stage GroupKFold Validation
# =========================================================

output_dir = "artifacts/reports/flowcharts"
os.makedirs(output_dir, exist_ok=True)

flow = Digraph("groupkfold_flowchart", format="png")
flow.attr(rankdir="TB", splines="ortho", nodesep="0.5", ranksep="0.7")
flow.attr("node", shape="rectangle", style="rounded,filled", fillcolor="#EAF4FF", color="#2F5D8A", fontname="Arial", fontsize="11")
flow.attr("edge", color="#2F5D8A", arrowsize="0.8")

# Title Node
flow.node("start", "Start: Additional 4-Stage GroupKFold Validation", shape="oval", fillcolor="#D9EFFF")

# Steps
flow.node("load", "Load Full Dataset\n(4920 Samples, 41 Diseases)")
flow.node("preprocess", "Apply Preprocessing\n- Normalize symptoms\n- Pad to length 17\n- Encode symptoms & severity\n- Create numerical inputs")
flow.node("patterns", "Create Canonical Order-Independent\nSymptom-Severity Patterns")
flow.node("groups", "Assign Group IDs\nfor Unique Symptom-Severity Patterns\n(Total Unique Groups = 305)")
flow.node("split", "Apply GroupKFold\n(n_splits = 4)")

# Stage Loop
flow.node("stage1", "Stage 1\nTrain + Validation + Test")
flow.node("stage2", "Stage 2\nTrain + Validation + Test")
flow.node("stage3", "Stage 3\nTrain + Validation + Test")
flow.node("stage4", "Stage 4\nTrain + Validation + Test")

flow.node("inside", "Inside Each Stage:\n- Ensure Zero Group Overlap\n- Train Enhanced BiLSTM\n- Evaluate on Test Fold\n- Record Accuracy, Precision,\nRecall, F1-score, Top-K")

flow.node("aggregate", "Aggregate Results Across 4 Stages\n- Mean Accuracy\n- Mean Macro Precision\n- Mean Macro Recall\n- Mean Macro F1\n- Mean Top-3 Accuracy")
flow.node("final", "Final Summary:\nMean Test Accuracy = 99.04%\nMean Macro F1 = 98.67%\nMean Top-3 Accuracy = 100%\nCorrect Predictions = 4873\nWrong Predictions = 47", fillcolor="#DFF5E1", color="#2E7D32")
flow.node("end", "End", shape="oval", fillcolor="#D9EFFF")

# Connections
flow.edge("start", "load")
flow.edge("load", "preprocess")
flow.edge("preprocess", "patterns")
flow.edge("patterns", "groups")
flow.edge("groups", "split")

flow.edge("split", "stage1")
flow.edge("split", "stage2")
flow.edge("split", "stage3")
flow.edge("split", "stage4")

flow.edge("stage1", "inside")
flow.edge("stage2", "inside")
flow.edge("stage3", "inside")
flow.edge("stage4", "inside")

flow.edge("inside", "aggregate")
flow.edge("aggregate", "final")
flow.edge("final", "end")

# Render
output_path = os.path.join(output_dir, "groupkfold_4stage_flowchart")
flow.render(output_path, cleanup=True)

print(f"Flowchart saved to: {output_path}.png")