# verify_nn_fr.py
import sys
import os
import torch
import h5py
import numpy as np

# ===================================================================
# 1. CONFIGURATION
# ===================================================================
# --- Set these flags to control which tests are run ---
RUN_LAYER_BY_LAYER_CHECK = True
RUN_INFERENCE_COMPARISON = True
USE_GPU = False  # Master switch for CPU/GPU

# ===================================================================
# 2. SETUP: PATHS, IMPORTS, AND DEVICE
# ===================================================================
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    print("🚀 Using GPU (CUDA)")
else:
    device = torch.device('cpu')
    print("💻 Using CPU")

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../nn'))

# Import all necessary models and functions
from net import TSPDGraphTransformerNetwork
from net_fr import TSPDGraphTransformerNetworkFlyingRange
from infer import predict_chainlet_length as legacy_predict_single, batch_prediction as legacy_predict_batch
from infer_fr import predict_chainlet_length as new_predict_single, batch_prediction as new_predict_batch

print(f"Running on device: {device}")
print("-" * 50)

# ===================================================================
# 3. DATA AND MODEL LOADING
# ===================================================================
def load_data_samples(file_path, num_samples=10):
    """Loads the first N samples from the HDF5 dataset."""
    samples = []
    with h5py.File(file_path, 'r') as f:
        keys = sorted(list(f.keys()))[:num_samples]
        for key in keys:
            g = f[key]
            samples.append({
                'dist': g['dist_mtx'][:], 'tour': g['tour'][:],
                'alpha': g['alpha'][()], 'scale': g['scaling_factor'][()],
                'frng': g['flying_range'][()]
            })
    print(samples[0])
    return samples

# --- Load Data ---
DATA_H5_PATH = os.path.join(script_dir, 'samples.h5')
data_samples = load_data_samples(DATA_H5_PATH)
print(f"✅ Loaded {len(data_samples)} data samples for high-level tests.")

# --- Load BOTH Models ---
# Legacy Model
legacy_model = TSPDGraphTransformerNetwork(in_channels=8, hidden_channels=8, out_channels=8, heads=4, beta=False, dropout=0.0, normalization="graph_norm", num_gat_layers=4, activation="elu", readout_type="attention")
LEGACY_CKPT_PATH = os.path.join(script_dir, '../nn', 'trained', 'neural_cost_predictor.pth')
legacy_sd_cpu = torch.load(LEGACY_CKPT_PATH, map_location='cpu')
if list(legacy_sd_cpu.keys())[0].startswith('module.'):
    legacy_sd_cpu = {k[7:]: v for k, v in legacy_sd_cpu.items()}
legacy_sd_device = {k: v.to(device) for k, v in legacy_sd_cpu.items()}
legacy_model.to(device)
legacy_model.load_state_dict(legacy_sd_device)
legacy_model.eval()
print("✅ Legacy model loaded.")

# New Fine-Tuned Model
NEW_CKPT_PATH = os.path.join(script_dir, 'trained', 'neural_cost_predictor_fr.pth')
new_model_config = {"phi_layers": 2, "phi_hidden": 64, "phi_activation": "leaky_relu", "phi_leaky_slope": 0.01}
new_model = TSPDGraphTransformerNetworkFlyingRange(dropout=0.0, edge_dim=3, **new_model_config)
new_sd_cpu = torch.load(NEW_CKPT_PATH, map_location='cpu')['model_state_dict']
new_sd_device = {k: v.to(device) for k, v in new_sd_cpu.items()}
new_model.to(device)
new_model.load_state_dict(new_sd_device)
new_model.eval()
print("✅ Fine-tuned model loaded.")
print("=" * 50)


# ===================================================================
# TEST 1: LOW-LEVEL LAYER-BY-LAYER IDENTITY CHECK
# ===================================================================
if RUN_LAYER_BY_LAYER_CHECK:
    print("\n\n🔥 TEST 1: Performing Low-Level Layer-by-Layer Identity Check...")
    
    # --- Parameter Verification ---
    print("\n-- Verifying legacy parameters...")
    mismatches = 0
    legacy_keys = legacy_model.cpu().state_dict().keys()
    new_sd_cpu_verify = new_model.cpu().state_dict()
    for key_legacy in legacy_keys:
        key_new = key_legacy.replace("transformer_layers", "convs").replace("norm_layers", "norms").replace("output_layer", "out_proj")
        if key_new not in new_sd_cpu_verify: mismatches += 1; continue
        tensor_legacy = legacy_model.cpu().state_dict()[key_legacy]
        tensor_new = new_sd_cpu_verify[key_new]
        if "lin_edge.weight" in key_legacy: tensor_new = tensor_new[:, :2]
        if not torch.allclose(tensor_legacy, tensor_new, atol=1e-8): mismatches += 1
    if mismatches == 0: print("✅ Success! All legacy parameters are perfectly contained.")
    else: print(f"🔥 Found {mismatches} mismatches.")
    
    # --- Data Preparation for this test ---
    torch.manual_seed(42)
    num_nodes, num_edges, in_channels = 20, 100, 8
    x = torch.randn(num_nodes, in_channels, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    edge_attr_2d = torch.randn(num_edges, 2, device=device)
    edge_attr_3d = torch.nn.functional.pad(edge_attr_2d, (0, 1), "constant", 0)
    print("\n-- ✅ Random graph data created for low-level test.")
    
    # --- Propagation and Comparison ---
    print("\n-- Performing layer-by-layer comparison...")
    x_legacy = x.clone()
    x_new = x.clone()
    g_f_zero = torch.tensor([0.0], device=device)
    
    def compare_and_print(step_name, legacy_tensor, new_tensor):
        diff = torch.abs(legacy_tensor - new_tensor)
        print(f"    - {step_name:<18} | Max difference: {torch.max(diff).item():.8g}")

    for i in range(5):
        print(f"--- Processing Layer {i} ---")
        x_legacy_conv = legacy_model.transformer_layers[i](x_legacy, edge_index, edge_attr_2d)
        x_new_conv = new_model.convs[i](x_new, edge_index, edge_attr_3d)
        compare_and_print("After Conv", x_legacy_conv, x_new_conv)
        
        x_legacy_norm = legacy_model.norm_layers[i](x_legacy_conv, batch)
        x_new_norm = new_model.norms[i](x_new_conv, batch)
        compare_and_print("After Norm", x_legacy_norm, x_new_norm)

        x_new_film = new_model.films[i](x_new_norm, g_f_zero, batch)
        compare_and_print("  (f=0 FiLM check)", x_new_norm, x_new_film)
        
        x_legacy = legacy_model.activation(x_legacy_norm)
        x_new = new_model.act(x_new_film)
        compare_and_print("After Activation", x_legacy, x_new)
        print("")
    print("-" * 50)


# ===================================================================
# TEST 2: HIGH-LEVEL INFERENCE FUNCTION COMPARISON
# ===================================================================
if RUN_INFERENCE_COMPARISON:
    print("\n\n🔥 TEST 2: Comparing High-Level Inference Functions (Legacy vs. New)...")
    
    # --- Sub-Test 2A: Compare Single Predictions ---
    print("\n-- Comparing Single Predictions (f=inf)...")
    legacy_single_preds_raw = []
    new_single_preds_raw = []
    for sample in data_samples:
        raw_ct = sample['dist'] * sample['scale']
        raw_cd = raw_ct / sample['alpha']
        
        # Predict with legacy model (buggy, returns scaled) and de-normalize
        pred_legacy_scaled = legacy_predict_single(legacy_model, device, sample['tour'], raw_ct, raw_cd, 8, sample['scale'])
        pred_legacy_raw = pred_legacy_scaled * sample['scale']
        legacy_single_preds_raw.append(pred_legacy_raw)
        
        # Predict with new model (buggy, returns scaled) and de-normalize
        pred_new_scaled = new_predict_single(new_model, device, sample['tour'], raw_ct, raw_cd, 8, sample['scale'], float('inf'))
        pred_new_raw = pred_new_scaled * sample['scale']
        new_single_preds_raw.append(pred_new_raw)

    legacy_single_preds_raw = np.array(legacy_single_preds_raw)
    new_single_preds_raw = np.array(new_single_preds_raw)
    max_diff_single = np.max(np.abs(legacy_single_preds_raw - new_single_preds_raw))
    print(f"✅ Comparison complete. Maximum Absolute Difference: {max_diff_single:.8g}")

    # --- Sub-Test 2B: Compare Batch Predictions ---
    print("\n-- Comparing Batch Predictions (f=inf)...")
    chainlets_legacy = [{"initial_route": s['tour'], "C_t": s['dist']*s['scale'], "C_d": (s['dist']*s['scale'])/s['alpha'], "alpha": s['alpha']} for s in data_samples]
    scales = [s['scale'] for s in data_samples]
    alpha = [s['alpha'] for s in data_samples]
    alpha = np.mean(alpha)
    chainlets_new = [{"initial_route": s['tour'], "C_t": s['dist']*s['scale'], "C_d": (s['dist']*s['scale'])/s['alpha'], "flying_range": float('inf'), "alpha": s['alpha']} for s in data_samples]

    legacy_batch_preds_raw = legacy_predict_batch(device, legacy_model, 8, chainlets_legacy, scales).cpu().numpy()
    new_batch_preds_raw = new_predict_batch(device, new_model, 8, chainlets_new, scales).cpu().numpy()
    max_diff_batch = np.max(np.abs(legacy_batch_preds_raw - new_batch_preds_raw))
    print(f"✅ Comparison complete. Maximum Absolute Difference: {max_diff_batch:.8g}")
    print("-" * 50)

print("\n\n🎉 Test suite finished.")