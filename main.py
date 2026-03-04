
import os
import sys
import ctypes
import time
import select
import termios
import tty
import numpy as np
import pandas as pd
import cv2
import random
import mujoco
from mujoco import viewer 

# Create Qt fonts directory for OpenCV
os.makedirs(os.path.join(os.path.dirname(cv2.__file__), 'qt', 'fonts'), exist_ok=True)
from collections import deque

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Pandas Import for Rolling Stats ---
import pandas as pd

# --- Haptics Imports ---
try:
    from pyOpenHaptics.hd_device import HapticDevice
    import pyOpenHaptics.hd as hd
    from pyOpenHaptics.hd_callback import hd_callback
    from dataclasses import dataclass, field
    HAPTICS_AVAILABLE = True
except ImportError:
    print("Warning: pyOpenHaptics not found. Running in simulation mode without haptics.")
    HAPTICS_AVAILABLE = False

# --- Matplotlib Imports ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from itertools import cycle

# =============================================================================
# 1. NEURAL NETWORK & AGENT DEFINITIONS
# =============================================================================

# ====== Positional Encoding ======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ====== Graph Attention Layer ======
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        Wh = self.W(h)
        batch_size, N, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1))).squeeze(-1)
        attention = torch.softmax(e, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

# ====== BiACT Policy with Graph Attention ======
class GraphAttentionBiACTPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_layers=3, chunk_size=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.gat = GraphAttentionLayer(d_model, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            padding = torch.zeros(batch_size, pad_len, x.size(2), device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len += pad_len

        chunks = x.view(batch_size, -1, self.chunk_size, x.size(2))
        b, num_chunks, csize, d = chunks.size()
        chunks_reshaped = chunks.view(b * num_chunks, csize, d)
        gat_out = self.gat(chunks_reshaped)
        chunk_embeds = gat_out.mean(dim=1).view(b, num_chunks, d)

        mask = torch.triu(torch.ones(num_chunks, num_chunks) * float('-inf'), diagonal=1).to(x.device)
        decoded = self.transformer_decoder(tgt=chunk_embeds, memory=chunk_embeds, tgt_mask=mask)
        pooled = decoded.mean(dim=1)
        return self.fc_out(pooled)

# ====== Meta-RL Agent with MAML ======
class MetaRLAgentWithMAML:
    def __init__(self, input_dim, output_dim, seq_len=20, inner_lr=1e-2, outer_lr=1e-3):
        self.policy = GraphAttentionBiACTPolicy(input_dim, output_dim)
        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=outer_lr)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 8
        self.seq_len = seq_len
        self.inner_lr = inner_lr
        self.losses = []

    def act(self, state_seq):
        self.policy.eval()
        with torch.no_grad():
            if len(state_seq.shape) == 2:
                x = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0)
            elif len(state_seq.shape) == 3:
                x = torch.tensor(state_seq, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected state_seq shape: {state_seq.shape}")

            output = self.policy(x).squeeze(0)
            return output.cpu().numpy()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def compute_loss(self, model, batch):
        s, a, r, ns, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(np.array(ns), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(1)

        q_pred = model(s)
        q_next = model(ns).detach().max(dim=1, keepdim=True)[0]
        target = r + self.gamma * q_next * (1 - d)
        return F.mse_loss(q_pred, target)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        support_batch = batch[:self.batch_size // 2]
        query_batch = batch[self.batch_size // 2:]

        inner_model = GraphAttentionBiACTPolicy(
            input_dim=self.policy.input_proj.in_features,
            output_dim=self.policy.fc_out.out_features
        )
        inner_model.load_state_dict(self.policy.state_dict())

        inner_loss = self.compute_loss(inner_model, support_batch)
        grads = torch.autograd.grad(inner_loss, inner_model.parameters(), create_graph=True)
        updated_params = {name: param - self.inner_lr * grad
                          for (name, param), grad in zip(inner_model.named_parameters(), grads)}

        for name, param in inner_model.named_parameters():
            param.data = updated_params[name].data.clone()

        query_loss = self.compute_loss(inner_model, query_batch)

        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        self.losses.append(query_loss.item())

# =============================================================================
# 2. HAPTICS SETUP
# =============================================================================

if HAPTICS_AVAILABLE:
    os.environ['GTDD_HOME'] = os.path.expanduser('~/.3dsystems')

    def find_haptics_library():
        likely_path = '/home/shafiq/haptics_ws/drivers/openhaptics_3.4-0-developer-edition-amd64/usr/lib'
        if os.path.exists(os.path.join(likely_path, 'libHD.so')):
            return likely_path
        system_libs = ['/usr/lib', '/usr/local/lib', '/usr/lib/x86_64-linux-gnu']
        for path in system_libs:
            if os.path.exists(os.path.join(path, 'libHD.so')):
                return path
        search_root = '/home/shafiq/haptics_ws'
        for root, dirs, files in os.walk(search_root):
            if 'libHD.so' in files:
                return root
        return None

    lib_path = find_haptics_library()

    if lib_path:
        print(f"Found Haptics library at: {lib_path}")
        os.environ['LD_LIBRARY_PATH'] = lib_path
        try:
            ctypes.CDLL(os.path.join(lib_path, 'libHD.so'))
            print("Haptics Library loaded successfully.")
        except Exception as e:
            print(f"Error loading haptics library: {e}")
            lib_path = None
    else:
        print("WARNING: Could not find libHD.so. Haptics disabled.")
        lib_path = None

    @dataclass
    class DeviceState:
        position: list = field(default_factory=list)
        joints: list = field(default_factory=list)
        gimbals: list = field(default_factory=list)
        full_joints: list = field(default_factory=list)
        btn_top: bool = False
        btn_bottom: bool = False
        force: list = field(default_factory=list)

    device_state = DeviceState()

    @hd_callback
    def state_callback():
        global device_state
        try:
            motors = hd.get_joints()
            device_state.joints = [motors[0], motors[1], motors[2]]
            gimbals = hd.get_gimbals()
            device_state.gimbals = [gimbals[0], gimbals[1], gimbals[2]]
            device_state.full_joints = device_state.joints + device_state.gimbals
            btn_mask = hd.get_buttons()
            device_state.btn_top = (btn_mask & 1) != 0
            device_state.btn_bottom = (btn_mask & 2) != 0
            device_state.force = [0, 0, 0]
            hd.set_force(device_state.force)
        except Exception as e:
            pass

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def clamp(v, mn, mx): return np.clip(v, mn, mx)

def cosine_similarity(a, b, eps=1e-8):
    a_norm = np.linalg.norm(a) + eps
    b_norm = np.linalg.norm(b) + eps
    return np.dot(a, b) / (a_norm * b_norm)

def get_ctrl_indices(model):
    # Note: Standard ALOHA XML order is Left (fl) then Right (fr)
    indices = [model.actuator(f"{side}_joint{i}").id for side in ("fl", "fr") for i in range(1, 9)]
    return indices

def sigmoid_ramp(x, start=100, slope=0.05):
    return 1 / (1 + np.exp(-slope * (x - start)))

# =============================================================================
# 4. MAIN SIMULATION LOOP
# =============================================================================

def main():
    XML_PATH = "/home/shafiq/Desktop/0ALOHA-ALL/mobile_aloha_sim-master/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"
    if not os.path.exists(XML_PATH):
        XML_PATH = "aloha_v1.xml" 
    
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        print(f"Model loaded from {XML_PATH}")
    except Exception as e:
        print(f"XML Error: {e}")
        return

    data = mujoco.MjData(model)

    renderer = None
    try:
        renderer = mujoco.Renderer(model)
        print("Renderer initialized.")
    except:
        print("Offscreen renderer not available.")

    ctrl_indices = get_ctrl_indices(model)
    
    ctrl_ranges = np.array([
        [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
        [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
    ])

    # --- Agent Setup ---
    input_dim = len(ctrl_indices) * 2 
    output_dim = len(ctrl_indices)
    seq_len = 20
    agent = MetaRLAgentWithMAML(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)

    # --- Haptics Config ---
    SCALE_J1 = 4.0
    SCALE_J2 = 4.0
    SCALE_J3 = 4.0
    SCALE_J4 = 2.0
    SCALE_J5 = 2.0
    SCALE_J6 = 2.0
    
    haptic_device = None
    if HAPTICS_AVAILABLE and lib_path:
        try:
            haptic_device = HapticDevice(callback=state_callback, scheduler_type="async")
            time.sleep(0.2)
            print("Haptic Device Connected")
        except Exception as e:
            print(f"Haptic Device Error: {e}")

    # --- Variables for Loop ---
    right_arm_state = np.zeros(8)
    left_arm_state = np.zeros(8)
    right_gripper_val = 0.04
    left_gripper_val = 0.04
    GRIPPER_MIN = 0.0
    GRIPPER_MAX = 0.0475
    GRIPPER_STEP = 0.001
    prev_haptic_joints = np.zeros(6)

    # --- Data Collection Lists ---
    rewards, joint_positions, latencies, compliance_factors, actions_history = [], [], [], [], []
    alignment_scores = []
    assist_weights = []
    
    # Lists for Figure 1, 7, 12, 13, 14, 15, 16, 17
    r_raw_pos_hist = [] 
    r_actual_pos_hist = [] 
    l_raw_pos_hist = [] 
    l_actual_pos_hist = []
    manual_control_history = [] # For Figure 14

    seq_buffer = deque(maxlen=seq_len)
    mujoco.mj_forward(model, data)
    for _ in range(seq_len):
        s = np.concatenate([data.qpos[ctrl_indices], data.qvel[ctrl_indices]])
        seq_buffer.append(s)

    step = 0
    manual_active = False
    last_manual_move_step = 0
    
    # Default to right arm
    active_arm = "right"

    print("\n=== Starting Simulation ===")
    print("Controls: 'r' for Right Arm, 'l' for Left Arm. ESC to Exit.")
    print("Index Mapping: 0-7 (Left), 8-15 (Right)")
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        
        with viewer.launch_passive(model, data) as v:
            running = True
            cam_right_id = model.camera("fr_dabai").id if model.camera("fr_dabai") else -1
            cam_left_id = model.camera("fl_dabai").id if model.camera("fl_dabai") else -1

            while running:
                start_time = time.time()

                # 1. Keyboard Input for Arm Switching
                if select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char == 'r' or char == 'R':
                        active_arm = "right"
                        print(">>> [MODE] Switched to RIGHT ARM control (Indices 8-15)")
                    elif char == 'l' or char == 'L':
                        active_arm = "left"
                        print(">>> [MODE] Switched to LEFT ARM control (Indices 0-7)")
                    elif char == '\x1b':
                        running = False

                qpos = data.qpos[ctrl_indices]
                qvel = data.qvel[ctrl_indices]
                
                raw_delta = np.zeros(6)
                h_joints = np.zeros(6)
                
                if haptic_device:
                    h_joints = np.array(device_state.full_joints)
                    raw_delta = h_joints - prev_haptic_joints
                    # Swap J4/J5 Deltas
                    temp_delta_3 = raw_delta[3]
                    raw_delta[3] = raw_delta[4] 
                    raw_delta[4] = temp_delta_3
                    prev_haptic_joints = h_joints
                
                if np.linalg.norm(raw_delta) > 1e-4:
                    manual_active = True
                    last_manual_move_step = step
                elif step - last_manual_move_step > 50:
                    manual_active = False

                # --- CONTROL LOGIC: STRICT ARM ISOLATION (FIXED MAPPING) ---
                manual_ctrl = np.zeros(16)

                if active_arm == "right":
                    # --- Right Arm is ACTIVE ---
                    right_arm_state[0] += raw_delta[0] * SCALE_J1
                    right_arm_state[1] += raw_delta[1] * SCALE_J2
                    right_arm_state[2] += raw_delta[2] * SCALE_J3
                    right_arm_state[3] += raw_delta[3] * SCALE_J4
                    right_arm_state[4] += raw_delta[4] * SCALE_J5
                    right_arm_state[5] += raw_delta[5] * SCALE_J6

                    if device_state.btn_top: right_gripper_val -= GRIPPER_STEP
                    elif device_state.btn_bottom: right_gripper_val += GRIPPER_STEP
                    right_gripper_val = clamp(right_gripper_val, GRIPPER_MIN, GRIPPER_MAX)
                    right_arm_state[6:] = right_gripper_val
                    
                    # Indices 0-7: LEFT ARM (Hold)
                    manual_ctrl[:6] = left_arm_state[:6]
                    manual_ctrl[6:8] = left_gripper_val
                    # Indices 8-15: RIGHT ARM (Update)
                    manual_ctrl[8:14] = right_arm_state[:6]
                    manual_ctrl[14:16] = right_gripper_val

                else: # active_arm == "left"
                    # --- Left Arm is ACTIVE ---
                    left_arm_state[0] += raw_delta[0] * SCALE_J1
                    left_arm_state[1] += raw_delta[1] * SCALE_J2
                    left_arm_state[2] += raw_delta[2] * SCALE_J3
                    left_arm_state[3] += raw_delta[3] * SCALE_J4
                    left_arm_state[4] += raw_delta[4] * SCALE_J5
                    left_arm_state[5] += raw_delta[5] * SCALE_J6

                    if device_state.btn_top: left_gripper_val -= GRIPPER_STEP
                    elif device_state.btn_bottom: left_gripper_val += GRIPPER_STEP
                    left_gripper_val = clamp(left_gripper_val, GRIPPER_MIN, GRIPPER_MAX)
                    left_arm_state[6:] = left_gripper_val

                    # Indices 0-7: LEFT ARM (Update)
                    manual_ctrl[:6] = left_arm_state[:6]
                    manual_ctrl[6:8] = left_gripper_val
                    # Indices 8-15: RIGHT ARM (Hold)
                    manual_ctrl[8:14] = right_arm_state[:6]
                    manual_ctrl[14:16] = right_gripper_val

                # RL Agent Inference
                current_state = np.concatenate([qpos, qvel])
                seq_buffer.append(current_state)
                state_seq = np.array(seq_buffer)
                state_seq = (state_seq - np.mean(state_seq)) / (np.std(state_seq) + 1e-8)

                agent_action = agent.act(state_seq)
                actions_history.append(agent_action.copy())

                # Metrics
                v_norm = np.linalg.norm(qvel)
                m_norm = np.linalg.norm(raw_delta)
                
                raw_compliance = np.exp(-0.5 * v_norm)
                manual_boost = 0.2 * np.tanh(2 * m_norm)
                compliance_factor = np.clip(raw_compliance + manual_boost, 0.01, 1.0)
                compliance_factors.append(compliance_factor)

                if active_arm == "right":
                    active_indices = slice(8, 16) 
                else:
                    active_indices = slice(0, 8)  
                
                alignment_score = cosine_similarity(manual_ctrl[active_indices], agent_action[active_indices])
                alignment_scores.append(alignment_score)

                alignment_score_clamped = max(0, alignment_score)
                assist_weight = alignment_score_clamped * sigmoid_ramp(step)
                assist_weights.append(assist_weight)

                if manual_active and step >= 100:
                    final_ctrl = manual_ctrl + assist_weight * agent_action
                else:
                    final_ctrl = manual_ctrl

                final_ctrl_clipped = np.clip(final_ctrl, ctrl_ranges[:, 0], ctrl_ranges[:, 1])
                data.ctrl[ctrl_indices] = final_ctrl_clipped
                
                mujoco.mj_step(model, data)

                # --- Collect History ---
                l_raw_pos_hist.append(left_arm_state[:6].copy())
                l_actual_pos_hist.append(data.qpos[ctrl_indices[:6]].copy())
                r_raw_pos_hist.append(right_arm_state[:6].copy())
                r_actual_pos_hist.append(data.qpos[ctrl_indices[8:14]].copy())
                manual_control_history.append(manual_ctrl.copy())

                # Rendering & Logging
                if renderer and (cam_right_id != -1 or cam_left_id != -1) and step % 2 == 0:
                    cam_id = cam_right_id if active_arm == "right" else cam_left_id
                    if cam_id != -1:
                        renderer.update_scene(data, camera=cam_id)
                        pixels = renderer.render()
                        if pixels is not None and pixels.size > 0:
                            img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                            cv2.putText(img, f"ARM: {active_arm.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow("Wrist Camera", img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                running = False

                reward = alignment_score_clamped - 0.01 * v_norm
                rewards.append(reward)
                joint_positions.append(final_ctrl_clipped.copy())
                
                next_qpos = data.qpos[ctrl_indices]
                next_qvel = data.qvel[ctrl_indices]
                next_single_state = np.concatenate([next_qpos, next_qvel])
                
                next_seq_list = list(seq_buffer)[1:] + [next_single_state]
                next_state_seq = np.array(next_seq_list)
                next_state_seq = (next_state_seq - np.mean(next_state_seq)) / (np.std(next_state_seq) + 1e-8)
                
                agent.remember(state_seq, final_ctrl_clipped, reward, next_state_seq, False)

                if step >= 20 and step % 20 == 0:
                    agent.train()

                latencies.append(time.time() - start_time)
                if step % 10 == 0:
                    print(f"[STEP {step}] Arm: {active_arm.upper()} | Reward: {reward:.4f} | Align: {alignment_score:.3f}")

                v.sync()
                step += 1

    except Exception as e:
        print(f"\nRuntime Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if haptic_device:
            haptic_device.close()
        cv2.destroyAllWindows()
        if renderer: renderer.close()
        print("\nSimulation finished. Generating Reports...")

    # =============================================================================
    # 5. PLOTTING (ALL 17 FIGURES)
    # =============================================================================

    if step == 0:
        print("No steps recorded. Skipping plot generation.")
        return

    steps = np.arange(len(r_raw_pos_hist)) 

    # --- FIGURE 1 ---
    print("Generating Figure 1: Smoothing Over Steps (Input vs Output)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("Hybrid Bi-Action Smoothing Analysis (Overall 6-DOF)", fontsize=20)
    ax1.set_xlabel("Time Steps", fontsize=20)
    ax1.set_ylabel("Overall Velocity Magnitude (L2 Norm)", fontsize=20)
    try:
        r_raw_pos_matrix = np.array(r_raw_pos_hist).T
        r_actual_pos_matrix = np.array(r_actual_pos_hist).T
        raw_vel_matrix = np.gradient(r_raw_pos_matrix, axis=1)
        smooth_vel_matrix = np.gradient(r_actual_pos_matrix, axis=1)
        raw_overall = np.linalg.norm(raw_vel_matrix, axis=0)
        smooth_overall = np.linalg.norm(smooth_vel_matrix, axis=0)
    except: raw_overall = np.zeros(len(steps)); smooth_overall = np.zeros(len(steps))
    if len(raw_overall) > 0:
        var_raw = np.var(raw_overall); var_smooth = np.var(smooth_overall)
        avg_raw = np.mean(raw_overall); avg_smooth = np.mean(smooth_overall)
        reduction_pct = ((var_raw - var_smooth) / var_raw * 100) if var_raw > 1e-9 else 0.0
        ax1.plot(steps[:len(raw_overall)], raw_overall, label=f"Raw Input (Avg: {avg_raw:.3f})", color='red', alpha=0.5, linestyle='--')
        ax1.plot(steps[:len(smooth_overall)], smooth_overall, label=f"Smoothed Output (Avg: {avg_smooth:.3f}, Red: {reduction_pct:.1f}%)", color='green', linewidth=2)
        ax1.legend(loc='upper right', fontsize=20); ax1.tick_params(axis='both', labelsize=20); ax1.grid(True, alpha=0.3)
        plt.savefig("1 Smoothing Analysis.pdf", format='pdf', bbox_inches='tight'); plt.show(); plt.close()


    # --- FIGURE 3: SMOOTHNESS (VELOCITY TIME SERIES) ---
    print("Generating Figure 3: Smoothness Comparison")
    fig12 = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig12)
    ax12a = fig12.add_subplot(gs[0, 0]); ax12b = fig12.add_subplot(gs[0, 1]); ax12c = fig12.add_subplot(gs[1, :])
    fig12.suptitle("Smoothness Analysis (Velocity Magnitude Time Series)", fontsize=15)

    def get_improvement_label(avg_raw, avg_proposed, label_name, metric_name="Avg"):
        imp = ((avg_raw - avg_proposed) / avg_raw * 100) if avg_raw > 1e-9 else 0.0
        return f"{label_name} ({metric_name}: {avg_proposed:.3f}, Red: {imp:.1f}%)"

    # Calculate Time Series of Velocity Magnitude (Speed)
    try:
        l_raw_pos_matrix = np.array(l_raw_pos_hist).T
        l_smooth_pos_matrix = np.array(l_actual_pos_hist).T
        # Raw Velocity = Gradient of Raw Position
        l_raw_vel_series = np.linalg.norm(np.gradient(l_raw_pos_matrix, axis=1), axis=0)
        # Proposed Velocity = Gradient of Actual Position
        l_smooth_vel_series = np.linalg.norm(np.gradient(l_smooth_pos_matrix, axis=1), axis=0)
        
        r_raw_pos_matrix = np.array(r_raw_pos_hist).T
        r_smooth_pos_matrix = np.array(r_actual_pos_hist).T
        r_raw_vel_series = np.linalg.norm(np.gradient(r_raw_pos_matrix, axis=1), axis=0)
        r_smooth_vel_series = np.linalg.norm(np.gradient(r_smooth_pos_matrix, axis=1), axis=0)

        avg_raw_vel_series = (l_raw_vel_series + r_raw_vel_series) / 2.0
        avg_smooth_vel_series = (l_smooth_vel_series + r_smooth_vel_series) / 2.0
    except:
        l_raw_vel_series = np.zeros(len(steps)); l_smooth_vel_series = np.zeros(len(steps))
        r_raw_vel_series = np.zeros(len(steps)); r_smooth_vel_series = np.zeros(len(steps))
        avg_raw_vel_series = np.zeros(len(steps)); avg_smooth_vel_series = np.zeros(len(steps))

    # Left Arm
    avg_l_raw = np.mean(l_raw_vel_series)
    avg_l_smooth = np.mean(l_smooth_vel_series)
    ax12a.plot(steps, l_raw_vel_series, label=f"Raw (Avg: {avg_l_raw:.3f})", color='red', alpha=0.5, linestyle='--')
    ax12a.plot(steps, l_smooth_vel_series, label=get_improvement_label(avg_l_raw, avg_l_smooth, "Proposed", "Avg"), color='green', linewidth=2)
    ax12a.set_title("Left Arm (6-DOF)", fontsize=15); ax12a.set_ylabel("Velocity Magnitude", fontsize=15)
    ax12a.legend(loc='upper right', fontsize=15); ax12a.grid(True, alpha=0.3); ax12a.set_ylim(bottom=0)

    # Right Arm
    avg_r_raw = np.mean(r_raw_vel_series)
    avg_r_smooth = np.mean(r_smooth_vel_series)
    ax12b.plot(steps, r_raw_vel_series, label=f"Raw (Avg: {avg_r_raw:.3f})", color='red', alpha=0.5, linestyle='--')
    ax12b.plot(steps, r_smooth_vel_series, label=get_improvement_label(avg_r_raw, avg_r_smooth, "Proposed", "Avg"), color='green', linewidth=2)
    ax12b.set_title("Right Arm (6-DOF)", fontsize=15); ax12b.legend(loc='upper right', fontsize=15); ax12b.grid(True, alpha=0.3); ax12b.set_ylim(bottom=0)

    # Average
    avg_raw_total = np.mean(avg_raw_vel_series)
    avg_smooth_total = np.mean(avg_smooth_vel_series)
    ax12c.plot(steps, avg_raw_vel_series, label=f"Raw (Avg: {avg_raw_total:.3f})", color='purple', alpha=0.5, linestyle='--')
    ax12c.plot(steps, avg_smooth_vel_series, label=get_improvement_label(avg_raw_total, avg_smooth_total, "Proposed", "Avg"), color='cyan', linewidth=2)
    ax12c.set_title("Average Smoothness (Left + Right Arms)", fontsize=15)
    ax12c.set_xlabel("Time Steps", fontsize=15); ax12c.legend(loc='upper right', fontsize=15); ax12c.grid(True, alpha=0.3); ax12c.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("3 SMOOTHNESS (VELOCITY TIME SERIES).pdf", format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()


    # --- FIGURE 4: VELOCITY COMPARISON (TIME SERIES) ---
    print("Generating Figure 4: Velocity Comparisons")
    
    # Define the helper function if not already defined globally
    def get_improvement_label(avg_raw, avg_proposed, label_name, metric_name="Avg"):
        imp = ((avg_raw - avg_proposed) / avg_raw * 100) if avg_raw > 1e-9 else 0.0
        return f"{label_name} ({metric_name}: {avg_proposed:.3f}, Red: {imp:.1f}%)"

    # Create Figure with 2x2 GridSpec layout
    fig13 = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig13)
    
    # Define subplots: Top-Left, Top-Right, Bottom-Spanning
    ax13_left = fig13.add_subplot(gs[0, 0])
    ax13_right = fig13.add_subplot(gs[0, 1])
    ax13_avg = fig13.add_subplot(gs[1, :])
    
    fig13.suptitle("Velocity Magnitude Comparison (Time Series)", fontsize=15)

    # Calculate Velocity Time Series
    # Ensure steps variable is defined (assuming l_raw_pos_hist length)
    steps = np.arange(len(l_raw_pos_hist)) if 'steps' not in locals() else steps
    
    l_vel_raw = np.linalg.norm(np.gradient(np.array(l_raw_pos_hist).T, axis=1), axis=0)
    l_vel_prop = np.linalg.norm(np.gradient(np.array(l_actual_pos_hist).T, axis=1), axis=0)
    r_vel_raw = np.linalg.norm(np.gradient(np.array(r_raw_pos_hist).T, axis=1), axis=0)
    r_vel_prop = np.linalg.norm(np.gradient(np.array(r_actual_pos_hist).T, axis=1), axis=0)
    avg_vel_raw = (l_vel_raw + r_vel_raw) / 2.0
    avg_vel_prop = (l_vel_prop + r_vel_prop) / 2.0

    # Left Arm Plot
    avg_l_raw = np.mean(l_vel_raw)
    avg_l_prop = np.mean(l_vel_prop)
    ax13_left.plot(steps, l_vel_raw, label=f"Raw (Avg: {avg_l_raw:.3f})", color='red', alpha=0.5, linestyle='--')
    ax13_left.plot(steps, l_vel_prop, label=get_improvement_label(avg_l_raw, avg_l_prop, "Proposed", "Avg"), color='green', linewidth=2)
    ax13_left.set_title("Left Arm (6-DOF)", fontsize=15)
    ax13_left.set_ylabel("Velocity Norm", fontsize=15)
    ax13_left.legend(loc='upper right', fontsize=15)
    ax13_left.grid(True, alpha=0.3)
    ax13_left.set_ylim(bottom=0)

    # Right Arm Plot
    avg_r_raw = np.mean(r_vel_raw)
    avg_r_prop = np.mean(r_vel_prop)
    ax13_right.plot(steps, r_vel_raw, label=f"Raw (Avg: {avg_r_raw:.3f})", color='red', alpha=0.5, linestyle='--')
    ax13_right.plot(steps, r_vel_prop, label=get_improvement_label(avg_r_raw, avg_r_prop, "Proposed", "Avg"), color='green', linewidth=2)
    ax13_right.set_title("Right Arm (6-DOF)", fontsize=15)
    ax13_right.legend(loc='upper right', fontsize=15)
    ax13_right.grid(True, alpha=0.3)
    ax13_right.set_ylim(bottom=0)

    # Average Plot
    avg_raw_total = np.mean(avg_vel_raw)
    avg_smooth_total = np.mean(avg_vel_prop)
    ax13_avg.plot(steps, avg_vel_raw, label=f"Raw (Avg: {avg_raw_total:.3f})", color='purple', alpha=0.5, linestyle='--')
    ax13_avg.plot(steps, avg_vel_prop, label=get_improvement_label(avg_raw_total, avg_smooth_total, "Proposed", "Avg"), color='cyan', linewidth=2)
    ax13_avg.set_title("Average Velocity (Left + Right Arms)", fontsize=15)
    ax13_avg.set_xlabel("Time Steps", fontsize=15)
    ax13_avg.legend(loc='upper right', fontsize=15)
    ax13_avg.grid(True, alpha=0.3)
    ax13_avg.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("4 Velocity Comparisons.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # --- FIGURE 5: FORCE (EFFORT) COMPARISON (TIME SERIES) ---
    print("Generating Figure 5: Force/Effort Control Comparisons")
    fig14, axs14 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig14.suptitle("Control Effort Comparison (Proxy for Force)", fontsize=15)
    
    man_hist = np.array(manual_control_history)
    final_hist = np.array(joint_positions)
    
    # Left Effort
    l_man_eff = np.linalg.norm(man_hist[:, :6], axis=1)
    l_fin_eff = np.linalg.norm(final_hist[:, :6], axis=1)
    # Right Effort
    r_man_eff = np.linalg.norm(man_hist[:, 8:14], axis=1)
    r_fin_eff = np.linalg.norm(final_hist[:, 8:14], axis=1)
    # Average Effort
    avg_man_eff = (l_man_eff + r_man_eff) / 2.0
    avg_fin_eff = (l_fin_eff + r_fin_eff) / 2.0

    # Left
    axs14[0].plot(steps, l_man_eff, label=f"Raw (Avg: {np.mean(l_man_eff):.3f})", color='red', alpha=0.5, linestyle='--')
    axs14[0].plot(steps, l_fin_eff, label=get_improvement_label(np.mean(l_man_eff), np.mean(l_fin_eff), "Proposed", "Avg"), color='green', linewidth=2)
    axs14[0].set_title("Left Arm Effort"); axs14[0].legend(loc='upper right', fontsize=15); axs14[0].grid(True, alpha=0.3); axs14[0].set_ylabel("Effort Norm")

    # Right
    axs14[1].plot(steps, r_man_eff, label=f"Raw (Avg: {np.mean(r_man_eff):.3f})", color='red', alpha=0.5, linestyle='--')
    axs14[1].plot(steps, r_fin_eff, label=get_improvement_label(np.mean(r_man_eff), np.mean(r_fin_eff), "Proposed", "Avg"), color='green', linewidth=2)
    axs14[1].set_title("Right Arm Effort"); axs14[1].legend(loc='upper right', fontsize=15); axs14[1].grid(True, alpha=0.3)

    # Average
    axs14[2].plot(steps, avg_man_eff, label=f"Raw (Avg: {np.mean(avg_man_eff):.3f})", color='purple', alpha=0.5, linestyle='--')
    axs14[2].plot(steps, avg_fin_eff, label=get_improvement_label(np.mean(avg_man_eff), np.mean(avg_fin_eff), "Proposed", "Avg"), color='cyan', linewidth=2)
    axs14[2].set_title("Average Effort"); axs14[2].legend(loc='upper right', fontsize=15); axs14[2].grid(True, alpha=0.3); axs14[2].set_xlabel("Time Steps")
    plt.tight_layout(); plt.savefig("5 Force effort comparison.pdf", format='pdf', bbox_inches='tight'); plt.show(); plt.close()


    # --- FIGURE 9 ---
    print("Generating Figure 9: Training Dynamics")
    losses = agent.losses
    window_size = 10
    
    # Calculate x-axis for losses (Training happens every 20 steps starting from step 20)
    total_sim_steps = len(rewards) # Using rewards length as the reference for total steps
    
    if len(losses) > 0:
        # Generate step indices corresponding to when training occurred (20, 40, 60, ...)
        loss_steps = np.arange(20, 20 + 20 * len(losses), 20)[:len(losses)]
    else:
        loss_steps = np.array([])

    if len(losses) >= window_size:
        smooth_loss = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        # Adjust x-axis for smooth_loss (valid mode reduces length)
        smooth_steps = loss_steps[window_size-1:]
    else:
        smooth_loss = losses
        smooth_steps = loss_steps

    fig5, axs5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle("Training Dynamics", fontsize=20)
    
    # Subplot A: Smoothed Loss (Top Left)
    axs5[0, 0].plot(smooth_steps, smooth_loss, label="Agent Smoothed Loss", color='blue')
    axs5[0, 0].set_xlabel("Step", fontsize=20); axs5[0, 0].set_ylabel("Smoothed Loss", fontsize=20); axs5[0, 0].legend(loc='upper right', fontsize=20); axs5[0, 0].tick_params(axis='both', labelsize=20)
    axs5[0, 0].set_xlim(0, total_sim_steps) # Align with full simulation duration

    # Subplot B: Raw Loss (Top Right)
    axs5[0, 1].plot(loss_steps, losses, label="Agent Raw Loss", color='orange')
    axs5[0, 1].set_xlabel("Step", fontsize=20); axs5[0, 1].set_ylabel("Raw Loss", fontsize=20); axs5[0, 1].legend(loc='upper right', fontsize=20); axs5[0, 1].tick_params(axis='both', labelsize=20)
    axs5[0, 1].set_xlim(0, total_sim_steps) # Align with full simulation duration

    # Subplot C: Rewards (Bottom Left)
    axs5[1, 0].plot(rewards, label="Reward per step", color='green')
    axs5[1, 0].set_xlabel("Step", fontsize=20); axs5[1, 0].set_ylabel("Reward", fontsize=20); axs5[1, 0].legend(loc='upper right', fontsize=20); axs5[1, 0].tick_params(axis='both', labelsize=20)
    axs5[1, 0].set_xlim(0, total_sim_steps)

    # Subplot D: Latency (Bottom Right)
    axs5[1, 1].plot(latencies, label="Latency per step", color='green')
    axs5[1, 1].set_xlabel("Step", fontsize=20); axs5[1, 1].set_ylabel("Latency (s)", fontsize=20); axs5[1, 1].legend(loc='upper right', fontsize=20); axs5[1, 1].tick_params(axis='both', labelsize=20)
    axs5[1, 1].set_xlim(0, total_sim_steps)
    
    fig5.subplots_adjust(wspace=0.3, hspace=0.3); plt.draw()
    plt.savefig("9 Training Dynamics.pdf", format='pdf', dpi=1000); plt.show()

    # --- FIGURE 10 ---
    print("Generating Figure 10: Joint Behavior Analysis")
    fig6, axs6 = plt.subplots(2, 2, figsize=(14, 10))
    fig6.suptitle("Joint Behavior Analysis", fontsize=20)
    if len(actions_history) > 1:
        axs6[0, 0].plot(np.abs(np.diff(actions_history, axis=0)).mean(axis=1), label="Action Smoothness", color='blue')
    axs6[0, 0].set_xlabel("Step", fontsize=20); axs6[0, 0].set_ylabel("Change in joint position Δqpos", fontsize=20); axs6[0, 0].legend(loc='upper right', fontsize=20); axs6[0, 0].tick_params(axis='both', labelsize=20)

    axs6[0, 1].bar(np.arange(len(joint_positions[0])), np.var(joint_positions, axis=0), color='brown', label='Joint Stability (Variance)')
    axs6[0, 1].set_xlabel("Joint Index", fontsize=20); axs6[0, 1].set_ylabel("Variance", fontsize=20); axs6[0, 1].legend(loc='upper right', fontsize=20); axs6[0, 1].tick_params(axis='both', labelsize=20)

    im = axs6[1, 0].imshow(np.array(joint_positions).T, aspect='auto', cmap='viridis', interpolation='none')
    fig6.colorbar(im, ax=axs6[1, 0])
    axs6[1, 0].set_xlabel("Timestep", fontsize=20); axs6[1, 0].set_ylabel("Joint Index", fontsize=20)
    legend_proxy = Line2D([0], [0], linestyle="none", c='black', marker='s', label="Heatmap Joint Positions")
    axs6[1, 0].legend(handles=[legend_proxy], loc='upper right', fontsize=20); axs6[1, 0].tick_params(axis='both', labelsize=20)

    axs6[1, 1].bar(np.arange(len(joint_positions[0])), np.max(joint_positions, axis=0) - np.min(joint_positions, axis=0), color='darkcyan', label='Joint Movement Range')
    axs6[1, 1].set_xlabel("Joint Index", fontsize=20); axs6[1, 1].set_ylabel("Movement Range", fontsize=20); axs6[1, 1].legend(loc='upper right', fontsize=20); axs6[1, 1].tick_params(axis='both', labelsize=20)
    fig6.subplots_adjust(wspace=0.3, hspace=0.3); plt.draw()
    plt.savefig("10 Joint Behavior Analysis.pdf", format='pdf', dpi=1000); plt.show()


    # --- FIGURE 11 ---
    print("Generating Figure 11: Policy Control Characteristics")
    fig8, axs8 = plt.subplots(2, 2, figsize=(14, 10))
    fig8.suptitle("Policy Control Characteristics", fontsize=20)
    axs8[0, 0].plot(compliance_factors, label="Compliance Factor Used", color='blue')
    axs8[0, 0].set_xlabel("Step", fontsize=20); axs8[0, 0].set_ylabel("Compliance Factor", fontsize=20); axs8[0, 0].legend(loc='upper right', fontsize=20); axs8[0, 0].tick_params(axis='both', labelsize=20)
    axs8[0, 1].plot(np.linalg.norm(joint_positions, axis=1), label="Joint Velocity Norm", color='green')
    axs8[0, 1].set_xlabel("Step", fontsize=20); axs8[0, 1].set_ylabel("Velocity Norm ||q̇||", fontsize=20); axs8[0, 1].legend(loc='upper right', fontsize=20); axs8[0, 1].tick_params(axis='both', labelsize=20)
    axs8[1, 0].plot(np.linalg.norm(np.diff(joint_positions, axis=0), axis=1), label="Δ Change in Joint Velocity", color='blue')
    axs8[1, 0].set_xlabel("Step", fontsize=20); axs8[1, 0].set_ylabel("Δ||q̇|| Velocity Norm ", fontsize=20); axs8[1, 0].legend(loc='upper right', fontsize=20); axs8[1, 0].tick_params(axis='both', labelsize=20)
    axs8[1, 1].plot(np.gradient(compliance_factors), label="ΔChange Compliance", color='green')
    axs8[1, 1].set_xlabel("Step", fontsize=20); axs8[1, 1].set_ylabel("Δ Compliance Factor", fontsize=20); axs8[1, 1].legend(loc='upper right', fontsize=20); axs8[1, 1].tick_params(axis='both', labelsize=20)

    fig8.subplots_adjust(wspace=0.3, hspace=0.3); plt.draw()
    plt.savefig("11 Policy Control Characteristics.pdf", format='pdf', dpi=1000); plt.show()

    # --- FIGURE 12 ---
    print("Generating Figure 12: Alignment Score")
    fig9 = plt.figure(figsize=(14, 7))
    plt.title("Agent Assisted Alignment Score Over Time", fontsize=27)
    alignment_scores_arr = np.array(alignment_scores)
    plt.plot(alignment_scores_arr, label="Alignment Score", color='blue', linestyle='-', linewidth=2.5, marker='*', markersize=4)
    plt.xlabel("Step", fontsize=27); plt.ylabel("Alignment Score (Dot Product)", fontsize=27); plt.legend(loc='upper right', fontsize=27); plt.xticks(fontsize=27); plt.yticks(fontsize=27)
    plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.draw()
    plt.savefig("12 Alignment Score.pdf", format='pdf', dpi=1000); plt.show()

    # --- FIGURE 13 ---
    print("Generating Figure 13: Agent Actions")
    line_styles = cycle(['-', '--', '-.', ':']); markers = cycle(['', 'o', 's', 'D', 'x', '^', '*', 'v'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("Agent Assisted Actions Over Time Step", fontsize=20); actions_arr = np.array(actions_history)
    ax1.set_title("Left Arm Joints", fontsize=20)
    for joint_idx in range(8):
        style = next(line_styles); marker = next(markers)
        ax1.plot(actions_arr[:, joint_idx], label=f"Joint {joint_idx + 1}", linestyle=style, marker=marker, linewidth=1, markersize=1)
    ax1.set_ylabel("Agent Action Value", fontsize=20); ax1.tick_params(axis='both', which='major', labelsize=20); ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=20, ncol=1, frameon=True)
    line_styles = cycle(['-', '--', '-.', ':']); markers = cycle(['', 'o', 's', 'D', 'x', '^', '*', 'v'])
    ax2.set_title("Right Arm Joints", fontsize=20)
    for joint_idx in range(8, 16):
        style = next(line_styles); marker = next(markers)
        ax2.plot(actions_arr[:, joint_idx], label=f"Joint {joint_idx + 1}", linestyle=style, marker=marker, linewidth=1, markersize=1)
    ax2.set_xlabel("Step", fontsize=20); ax2.set_ylabel("Agent Action Value", fontsize=20); ax2.tick_params(axis='both', which='major', labelsize=20); ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=20, ncol=1, frameon=True)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95]); plt.subplots_adjust(top=0.9); plt.savefig("13 Agent Actions.pdf", format='pdf', dpi=1000, bbox_inches='tight'); plt.show()


    # --- FIGURE 14 ---
    print("Generating Figure 14: Summary Metrics")
    
    # Set global font size to 20 for all text elements (titles, labels, ticks, legends)
    plt.rcParams.update({'font.size': 20})
    
    fig11, axs11 = plt.subplots(2, 2, figsize=(14, 10))
    fig11.suptitle("Summary Metrics: Rewards, Compliance, Action Intensity, Alignment")
    
    axs11[0, 0].plot(rewards, color='green', label='Reward'); axs11[0, 0].set_title("Rewards per Step"); axs11[0, 0].set_xlabel("Step"); axs11[0, 0].set_ylabel("Reward"); axs11[0, 0].legend(loc='upper right'); axs11[0, 0].grid(True, alpha=0.3)
    
    axs11[0, 1].plot(compliance_factors, color='blue', label='Compliance Factor'); axs11[0, 1].set_title("Compliance Factor per Step"); axs11[0, 1].set_xlabel("Step"); axs11[0, 1].set_ylabel("Compliance Factor"); axs11[0, 1].legend(loc='upper right'); axs11[0, 1].grid(True, alpha=0.3)
    
    if len(actions_history) > 0:
        actions_arr = np.array(actions_history); action_norms = np.linalg.norm(actions_arr, axis=1)
        axs11[1, 0].plot(action_norms, color='purple', label='Action Norm')
    axs11[1, 0].set_title("Agent Action Intensity per Step"); axs11[1, 0].set_xlabel("Step"); axs11[1, 0].set_ylabel("Action Magnitude"); axs11[1, 0].legend(loc='upper right'); axs11[1, 0].grid(True, alpha=0.3)

    axs11[1, 1].plot(alignment_scores, color='red', label='Alignment Score'); axs11[1, 1].set_title("Agent Assisted Alignment Score per Step"); axs11[1, 1].set_xlabel("Step"); axs11[1, 1].set_ylabel("Alignment Score"); axs11[1, 1].legend(loc='upper right'); axs11[1, 1].grid(True, alpha=0.3)
    
    fig11.subplots_adjust(wspace=0.3, hspace=0.3); plt.draw()
    plt.savefig("14 Summary Metrics.pdf", format='pdf', bbox_inches='tight'); plt.show(); plt.close()
if __name__ == "__main__":
    main()
