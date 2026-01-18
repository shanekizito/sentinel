import torch
import os
from models.gcn_v2 import LogicEmbeddingModel
from models.reflex_transformer import RemediationModel

def export_sovereign_models(output_dir="./export"):
    """
    Exports neural models to optimized formats (TorchScript) for production mesh.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Sovereign Deployment: Exporting models to {output_dir}...")

    # 1. Export CPG GCN Encoder
    gcn_model = LogicEmbeddingModel()
    gcn_model.eval()
    
    # Tracing the GCN requires dummy inputs
    dummy_x = torch.randn(10, 128)
    dummy_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    dummy_batch = torch.zeros(10, dtype=torch.long)
    
    traced_gcn = torch.jit.trace(gcn_model, (dummy_x, dummy_edge_index, dummy_batch))
    traced_gcn.save(f"{output_dir}/cpg_gcn_v2.pt")
    print("  [v] GCN Encoder: Exported to TorchScript.")

    # 2. Export Reflex Transformer
    remediation_model = RemediationModel(vocab_size=10000)
    remediation_model.eval()
    
    dummy_tgt = torch.randint(0, 10000, (5, 1))
    dummy_logic_ctx = torch.randn(1, 512)
    
    traced_reflex = torch.jit.trace(remediation_model, (dummy_tgt, dummy_logic_ctx))
    traced_reflex.save(f"{output_dir}/reflex_v1.pt")
    print("  [v] Reflex Transformer: Exported to TorchScript.")

def generate_triton_config(output_dir="./triton_repository"):
    """
    Generates the industrial config.pbtxt for NVIDIA Triton Inference Server.
    """
    gcn_config = """
name: "cpg_gcn_v2"
platform: "pytorch_libtorch"
max_batch_size: 128
input [
  { name: "x", data_type: TYPE_FP32, dims: [ -1, 128 ] },
  { name: "edge_index", data_type: TYPE_INT64, dims: [ 2, -1 ] },
  { name: "batch", data_type: TYPE_INT64, dims: [ -1 ] }
]
output [
  { name: "embedding", data_type: TYPE_FP32, dims: [ 512 ] }
]
instance_group [ { count: 4, kind: KIND_GPU } ]
"""
    
    if not os.path.exists(f"{output_dir}/cpg_gcn_v2"):
        os.makedirs(f"{output_dir}/cpg_gcn_v2")
    
    with open(f"{output_dir}/cpg_gcn_v2/config.pbtxt", "w") as f:
        f.write(gcn_config)
    
    print("Sovereign Deployment: Triton configuration manifests generated.")

def verify_deployment_artifacts(export_dir="./export"):
    """
    Verifies the integrity of the exported sovereign neural artifacts.
    """
    gcn_path = f"{export_dir}/cpg_gcn_v2.pt"
    if os.path.exists(gcn_path):
        loaded_model = torch.jit.load(gcn_path)
        print("Sovereign Deployment Verification: GCN Artifact LOADED successfully.")
    else:
        print("Sovereign Deployment Verification: FAILED - GCN Artifact missing.")

if __name__ == "__main__":
    export_sovereign_models()
    generate_triton_config()
    verify_deployment_artifacts()
