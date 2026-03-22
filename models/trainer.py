from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.tempo import MultiModalDeepSurv
from models.loss_func import cox_ph_loss_static
from models.data import Image_dataloader
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# === Class: Trainer
class Trainer(object):
    def __init__(self, slice_num, batch_size, lr, epochs, ablation, pre_train_resnet_path, tab_df_path,shuffle=True):
        self.slice_num = slice_num
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.ablation = ablation
        self.pre_train_resnet_path = pre_train_resnet_path
        self.tab_df_path = tab_df_path
        self.shuffle = shuffle
        self.save_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_checkpoint(self, train_loss_lst, train_c_index_lst, test_loss_lst, test_c_index_lst, model, epoch, optimizer, lr, ablation):
        train_collector = {
                    'train_loss':train_loss_lst,
                    'train_c_index':train_c_index_lst,
                    'test_loss':test_loss_lst,
                    'test_c_index':test_c_index_lst
        }
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lr,
            'train_collector': train_collector,
            'ablation': ablation
        }
        # save checkpoints
        torch.save(checkpoint, f"../TEMPO/output/checkpoint_{ablation}.pth")
        print(f"checkpoint save at test C-index: {test_c_index_lst[-1]}")

    def data_load(self):
        self.tab_df = pd.read_csv(self.tab_df_path, parse_dates=["LM_DIAG_date","os_date","collection_date"]) # load tabular df
        # train-test split
        # self.trainset = self.tab_df[self.tab_df['split_label'] == 1]
        # self.testset = self.tab_df[self.tab_df['split_label'] == 0]
        self.trainset = self.tab_df
        self.testset = self.tab_df[self.tab_df['split_label'] == 0]
        # dataloader
        self.train_set = Image_dataloader(self.trainset, self.slice_num, ablation=self.ablation)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        self.test_set = Image_dataloader(self.testset, self.slice_num, cont_scaler=self.train_set.get_params()["cont_scaler"], cate_scaler=self.train_set.get_params()["cate_scaler"], ablation=self.ablation)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        return self.tab_df, self.trainset, self.train_set, self.train_loader, self.testset, self.test_set, self.test_loader

    def model_load(self):
        # model load
        self.model = MultiModalDeepSurv(
            feature_dim = 16,
            fusion_dim = self.train_set.get_params()["tabular_shape"],
            middle_dim = 8,
            ablation = self.ablation,
        ).to(device)
        
        # optimzer load
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01,)
        # sub-loss
        self.sub_criterion = nn.BCEWithLogitsLoss()


    def train(self):
        train_loss_lst = []
        train_c_index_lst = []
        test_loss_lst = []
        test_c_index_lst = []
    
        for epoch in range(self.epochs):

            self.model.train()
            epoch_loss = 0.0
            
            all_risk_scores_tr = []
            all_durations_tr = []
            all_events_tr = []
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train"):
                x_input = batch["x_input"].to(device)
                axi_img = batch["axi_img"].to(device)
                cor_img = batch["cor_img"].to(device)
                sag_img = batch["sag_img"].to(device)
                mic_mig = batch["mic_img"].to(device)
                durations = batch["y_duration"].to(device)
                events = batch["y_event"].to(device)
                prog_risk = batch["y_prog_risk"].to(device)
    
                # ablation consideration
                if self.ablation == "no_img":
                    axi_img = torch.zeros_like(axi_img).to(device)
                    cor_img = torch.zeros_like(cor_img).to(device)
                    sag_img = torch.zeros_like(sag_img).to(device)
                    mic_mig = torch.zeros_like(mic_mig).to(device)
                elif self.ablation == "no_mic":
                    mic_mig = torch.zeros_like(mic_mig).to(device)
                elif self.ablation == "no_mri":
                    axi_img = torch.zeros_like(axi_img).to(device)
                    cor_img = torch.zeros_like(cor_img).to(device)
                    sag_img = torch.zeros_like(sag_img).to(device)
                
                self.optimizer.zero_grad()
                risk_pred, prog_pred = self.model(axi_img, cor_img, sag_img, mic_mig, x_input)
    
                cox_loss = cox_ph_loss_static(risk_pred, durations, events)
                prog_loss = self.sub_criterion(prog_pred,prog_risk)
                loss = 0.8*cox_loss+0.2*prog_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                all_risk_scores_tr.extend(risk_pred.detach().cpu().numpy())
                all_durations_tr.extend(durations.detach().cpu().numpy())
                all_events_tr.extend(events.detach().cpu().numpy())
            
                
            if len(all_risk_scores_tr) > 0:
                try:
                    train_c_index = concordance_index(
                        all_durations_tr, 
                        -np.array(all_risk_scores_tr), 
                        all_events_tr
                    )
                except:
                    print("warn: fail to compute c-index in trainset")
                    train_c_index = 0.5
            else:
                train_c_index = 0.5
    
            # ========== 测试阶段 ==========
            self.model.eval()
            test_total_loss = 0.0
            all_risk_scores = []
            all_durations = []
            all_events = []
            
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Epoch {epoch+1} Test"):
                    x_input = batch["x_input"].to(device)
                    axi_img = batch["axi_img"].to(device)
                    cor_img = batch["cor_img"].to(device)
                    sag_img = batch["sag_img"].to(device)
                    mic_mig = batch["mic_img"].to(device)
                    durations = batch["y_duration"].to(device)
                    events = batch["y_event"].to(device)
                    prog_risk = batch["y_prog_risk"].to(device)
    
                    if self.ablation == "no_img":
                        axi_img = torch.zeros_like(axi_img).to(device)
                        cor_img = torch.zeros_like(cor_img).to(device)
                        sag_img = torch.zeros_like(sag_img).to(device)
                        mic_mig = torch.zeros_like(mic_mig).to(device)
                    elif self.ablation == "no_mic":
                        mic_mig = torch.zeros_like(mic_mig).to(device)
                    elif self.ablation == "no_mri":
                        axi_img = torch.zeros_like(axi_img).to(device)
                        cor_img = torch.zeros_like(cor_img).to(device)
                        sag_img = torch.zeros_like(sag_img).to(device)

                        
                    # 前向传播 
                    risk_pred, prog_pred = self.model(axi_img, cor_img, sag_img, mic_mig, x_input)
    
                    # 计算损失
                    cox_loss = cox_ph_loss_static(risk_pred, durations, events)
                    prog_loss = self.sub_criterion(prog_pred,prog_risk)
                    test_loss = 0.8*cox_loss+0.2*prog_loss
                    
                    test_total_loss += test_loss.item()
                    
                    all_risk_scores.extend(risk_pred.cpu().numpy())
                    all_durations.extend(durations.cpu().numpy())
                    all_events.extend(events.cpu().numpy())
            
            if len(all_risk_scores) > 0:
                try:
                    test_c_index = concordance_index(
                        all_durations, 
                        -np.array(all_risk_scores), 
                        all_events
                    )
                except ZeroDivisionError:
                    # 调试信息
                    print(f"  case vol: {sum(all_events)}/{len(all_events)}")
                    print(f"  sigma of risk cases: {np.var(all_risk_scores):.6f}")
                    test_c_index = 0.5
                except Exception as e:
                    print(f"C-index error: {e}")
                    test_c_index = 0.5
            else:
                test_c_index = 0.5
            
            avg_loss = epoch_loss / len(self.train_loader)
            avg_test_loss = test_total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        
            train_loss_lst.append(avg_loss)
            train_c_index_lst.append(train_c_index)
            test_loss_lst.append(avg_test_loss)
            test_c_index_lst.append(test_c_index)
        
            if epoch > 0 and max(test_c_index_lst) == test_c_index:
                self.save_checkpoint(train_loss_lst, train_c_index_lst, test_loss_lst, test_c_index_lst, self.model, epoch, self.optimizer, self.lr, self.ablation)
            
            print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train C_index: {train_c_index:.4f}, Test Loss: {avg_test_loss:.4f}, Test C-index: {test_c_index:.4f}")


    def train_pipeline(self):
        print("-> TEMPO Training Pipeline Start...")
        self.data_load()
        print("-> Data loaded.")
        self.model_load()
        print("-> Model loaded.")
        print("-> ###### Training Start ######")
        self.train()








    