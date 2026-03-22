import os
import gc
import warnings
import cv2
import torch
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.tempo import MultiModalDeepSurv
from models.trainer import Trainer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



slice_num = 5
batch_size = 32
lr = 0.0005
epochs = 50
ablation = "no_mic"
pre_train_resnet_path = "../TEMPO/dataset/MRI_pretrained.pth"
tab_df_path = "../TEMPO/dataset/tabular_input.csv"
checkpoint_path = "output/checkpoint_no_mic.pth"

if __name__ == "__main__":

    tempo = Trainer(slice_num, batch_size, lr, epochs, ablation, pre_train_resnet_path, tab_df_path, False)
    tab_df, trainset, train_set, trainloader, testset, test_set, testloader = tempo.data_load()
    model = MultiModalDeepSurv(
        feature_dim = 16,
                fusion_dim = train_set.get_params()["tabular_shape"],
                middle_dim = 8
    ).to(device)
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    class MultiModalCamWrapper(torch.nn.Module):
        def __init__(self, full_model, all_inputs, mode='axi'):
            super().__init__()
            self.full_model = full_model
            self.mode = mode
            self.axi, self.cor, self.sag, self.mic, self.x = all_inputs
    
        def forward(self, x_incoming):
            axi = x_incoming if self.mode == 'axi' else self.axi
            cor = x_incoming if self.mode == 'cor' else self.cor
            sag = x_incoming if self.mode == 'sag' else self.sag
            mic = x_incoming if self.mode == 'mic' else self.mic
            risk_score, _ = self.full_model(axi, cor, sag, mic, self.x)
            return risk_score.view(-1, 1)
    
    
    row_names = ["Original", "L1-last", "L2-last", "L3-last", "L4-last", 
                 "L1-pen", "L2-pen", "L3-pen", "L4-pen"]
    modes = ['axi', 'cor', 'sag', 'mic']
    target_layers_list = [
        model.encoder1.model.layer1[-1].conv2,
        model.encoder1.model.layer2[-1].conv2,
        model.encoder1.model.layer3[-1].conv2,
        model.encoder1.model.layer4[-1].conv2,
        model.encoder1.model.layer1[-2].conv2,
        model.encoder1.model.layer2[-2].conv2,
        model.encoder1.model.layer3[-2].conv2,
        model.encoder1.model.layer4[-2].conv2,
    
        model.encoder2.model.layer1[-1].conv2,
        model.encoder2.model.layer2[-1].conv2,
        model.encoder2.model.layer3[-1].conv2,
        model.encoder2.model.layer4[-1].conv2,
        model.encoder2.model.layer1[-2].conv2,
        model.encoder2.model.layer2[-2].conv2,
        model.encoder2.model.layer3[-2].conv2,
        model.encoder2.model.layer4[-2].conv2,
    
        model.encoder3.model.layer1[-1].conv2,
        model.encoder3.model.layer2[-1].conv2,
        model.encoder3.model.layer3[-1].conv2,
        model.encoder3.model.layer4[-1].conv2,
        model.encoder3.model.layer1[-2].conv2,
        model.encoder3.model.layer2[-2].conv2,
        model.encoder3.model.layer3[-2].conv2,
        model.encoder3.model.layer4[-2].conv2,
    
        model.encoder4.resnet.layer1[-1].conv2,
        model.encoder4.resnet.layer2[-1].conv2,
        model.encoder4.resnet.layer3[-1].conv2,
        model.encoder4.resnet.layer4[-1].conv2,
        model.encoder4.resnet.layer1[-2].conv2,
        model.encoder4.resnet.layer2[-2].conv2,
        model.encoder4.resnet.layer3[-2].conv2,
        model.encoder4.resnet.layer4[-2].conv2,
    ]
    
    
    print("Started...")

    total_samples = 0
    for batch in trainloader:
        total_samples += batch["x_input"].shape[0]

    pbar = tqdm.tqdm(total=total_samples, desc="Processing CAM visualizations")

    
    for batch in trainloader:
        x_input = batch["x_input"].to(device)
        y_duration = batch["y_duration"].to(device)
        y_event = batch["y_event"].to(device)
        axi_img = batch["axi_img"].to(device)
        cor_img = batch["cor_img"].to(device)
        sag_img = batch["sag_img"].to(device)
        mic_img = batch["mic_img"].to(device)
        patient_id = batch["patient_id"]
        
        for idx in range(x_input.shape[0]):
            pbar.update(1)
            axi_current = batch["axi_img"][idx:idx+1]
            cor_current = batch["cor_img"][idx:idx+1]
            sag_current = batch["sag_img"][idx:idx+1]
            mic_current = batch["mic_img"][idx:idx+1]
            def is_all_zero_strict(tensor):
                return torch.all(tensor == 0).item()
            skip_sample = False
            
            if ablation == "no_mic":
                is_zero_axi = is_all_zero_strict(axi_current)
                is_zero_cor = is_all_zero_strict(cor_current)
                is_zero_sag = is_all_zero_strict(sag_current)
                skip_sample = is_zero_axi and is_zero_cor and is_zero_sag
                prefix = "cam_mri"
                
            elif ablation == "no_mri":
                is_zero_mic = is_all_zero_strict(mic_current)
                skip_sample = is_zero_mic
                prefix = "cam_mic"

            elif ablation == "Full":
                is_zero_axi = is_all_zero_strict(axi_current)
                is_zero_cor = is_all_zero_strict(cor_current)
                is_zero_sag = is_all_zero_strict(sag_current)
                is_zero_mic = is_all_zero_strict(mic_current)
                skip_sample = is_zero_axi and is_zero_cor and is_zero_sag and is_zero_mic
                prefix = "cam_full"

            if skip_sample:
                pbar.set_postfix({"skipped": True})
                continue

            all_inputs = [
                axi_current.to(device),
                cor_current.to(device),
                sag_current.to(device),
                mic_current.to(device),
                batch["x_input"][idx:idx+1].to(device)
            ]

            fig, axes = plt.subplots(9, 4, figsize=(16, 24))
            
            for i, mode in enumerate(modes):
                img_slice = all_inputs[i][0, 0, :, :].cpu().detach().numpy()
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
                rgb_img = np.stack([img_slice] * 3, axis=-1).astype(np.float32)
                
                axes[0, i].imshow(rgb_img)
                axes[0, i].set_title(f"{mode.upper()} Original")
                axes[0, i].axis('off')
                for row_idx in range(1, 9):
                    # 计算 target_layers_list 中的索引
                    layer_idx_in_list = i * 8 + (row_idx - 1)
                    target_layer = target_layers_list[layer_idx_in_list]
                    wrapped_model = MultiModalCamWrapper(model, all_inputs, mode=mode).eval()
                    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
                    targets = [ClassifierOutputTarget(0)]
                    grayscale_cam = cam(input_tensor=all_inputs[i], targets=targets)[0, :]
                    vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    if i == 3:
                        vis_uint8 = (vis * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
                        mask = (grayscale_cam > 0.2).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_uint8, contours, -1, (255, 0, 0), 2)
                        vis = vis_uint8

                    axes[row_idx, i].imshow(vis)
                    if i == 0:
                        axes[row_idx, i].set_ylabel(row_names[row_idx], rotation=0, labelpad=40, 
                                                   fontsize=12, fontweight='bold')
                    
                    axes[row_idx, i].axis('off')
                    del cam, wrapped_model
                    torch.cuda.empty_cache()

            plt.tight_layout()
            save_path = f"cam_outputs/mri/{prefix}_9x4_{patient_id[idx]}_{idx}.png"
            os.makedirs("cam_outputs/mri", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            gc.collect()

            pbar.set_postfix({
                "patient_id": patient_id[idx],
                "saved": True,
                "file": save_path
            })

    pbar.close()
    print("Completed")
        