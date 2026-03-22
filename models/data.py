import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import h5py
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Image_dataloader(Dataset):
    def __init__(self, tabular_data, target_slice, cont_scaler=None, cate_scaler=None, ablation="Full", pretrain_resnet_path=None,):
        self.tabular_data = tabular_data
        if ablation == "modal_full":
            self.tabular_data = self.tabular_data[self.tabular_data["modal_check"]==1]
        del self.tabular_data["modal_check"]
        self.ablation = ablation
        patient_id = self.tabular_data.id.tolist()
        c_date = self.tabular_data.collection_date.tolist()
        if ablation == "no_img":
            axi_img = []
            cor_img = []
            sag_img = []
            mic_img = []
            for idx in range(len(patient_id)):
                axi_img.append(np.zeros((target_slice, 256, 256)))
                cor_img.append(np.zeros((target_slice, 256, 256)))
                sag_img.append(np.zeros((target_slice, 256, 256)))
                mic_img.append(np.zeros((1, 256, 256)))
        else:    
            # prepare img_data
            axi_img = []
            cor_img = []
            sag_img = []
            mic_img = []
    
            for idx in range(len(patient_id)):
                output = self.img_input_return(patient_id[idx], c_date[idx], target_slices=target_slice)
                # modal append
                axi_img.append(output['axi_img'])
                cor_img.append(output['cor_img'])
                sag_img.append(output['sag_img'])
                mic_img.append(output['mic_img'])
    
        axi_img = np.array(axi_img)
        cor_img = np.array(cor_img)
        sag_img = np.array(sag_img)
        mic_img = np.array(mic_img)

        # categorial features
        self.categorical_cols, self.continuous_cols = self.find_categorical_columns(self.tabular_data)
        if self.ablation == "no_time":
            self.continuous_cols.remove("time_len")

        # tabular x inputs
        x_continuous = self.tabular_data[self.continuous_cols].values
        x_categorical = self.tabular_data[self.categorical_cols].values
        # patient id
        self.pt_id = self.tabular_data['id'].values
        # y labels
        y_duration = self.tabular_data['duration'].values
        y_event = self.tabular_data['os_status'].values
        y_prog_risk = self.tabular_data['prog_risk'].values

        if cont_scaler is None:
            self.cont_scaler = StandardScaler()
            self.cate_scaler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            x_continuous = self.cont_scaler.fit_transform(x_continuous)
            x_categorical = self.cate_scaler.fit_transform(x_categorical)
            self.x_continuous = torch.FloatTensor(x_continuous)
            self.x_categorical = torch.FloatTensor(x_categorical)
            self.x_input = torch.cat([self.x_continuous, self.x_categorical], dim=1)
            # loss and labels
            self.y_duration = torch.FloatTensor(y_duration)
            self.y_event = torch.FloatTensor(y_event)
            self.y_prog_risk = torch.FloatTensor(y_prog_risk)
            # image inputs
            self.axi_img = torch.FloatTensor(axi_img)
            self.cor_img = torch.FloatTensor(cor_img)
            self.sag_img = torch.FloatTensor(sag_img)
            self.mic_img = torch.FloatTensor(mic_img)
        else:
            self.cont_scaler = cont_scaler
            self.cate_scaler = cate_scaler
            x_continuous = self.cont_scaler.transform(x_continuous)
            x_categorical = self.cate_scaler.transform(x_categorical)
            self.x_continuous = torch.FloatTensor(x_continuous)
            self.x_categorical = torch.FloatTensor(x_categorical)
            self.x_input = torch.cat([self.x_continuous, self.x_categorical], dim=1)
            # loss and labels
            self.y_duration = torch.FloatTensor(y_duration)
            self.y_event = torch.FloatTensor(y_event)
            self.y_prog_risk = torch.FloatTensor(y_prog_risk)
            # image inputs
            self.axi_img = torch.FloatTensor(axi_img)
            self.cor_img = torch.FloatTensor(cor_img)
            self.sag_img = torch.FloatTensor(sag_img)
            self.mic_img = torch.FloatTensor(mic_img)



    def get_params(self):
        return {
            "cont_scaler": self.cont_scaler,
            "cate_scaler": self.cate_scaler,
            "continuous_features": self.continuous_cols,
            "categorical_features": self.categorical_cols,
            "tabular_shape": self.x_input.shape[1],
            "mri_channels": self.axi_img.shape[1],
            "mri_size": self.axi_img.shape[2],
            "mic_channels": self.mic_img.shape[1],
            "mic_size": self.mic_img.shape[2],
            "tabular_data": self.tabular_data,
        }


    def find_categorical_columns(self, df, unique_threshold=6):
        categorical_cols = []
        continuous_cols = []
        

        for col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isna().sum()
            dtype = df[col].dtype
            
            # 判断逻辑
            if unique_count <= unique_threshold:
                categorical_cols.append(col)
                col_type = "分类"
            else:
                continuous_cols.append(col)
                col_type = "连续"

        cate_to_remove = ['split_label', 'prog_risk', 'os_status']
        cont_to_remove = ['id', 'os_date', 'LM_DIAG_date', 'collection_date', 'duration']

        categorical_cols = [x for x in categorical_cols if x not in cate_to_remove]
        continuous_cols = [x for x in continuous_cols if x not in cont_to_remove]
        
        return categorical_cols, continuous_cols
    

    def img_input_return(self, pt_id, collection_date, target_slices=30):
        axi_array = np.zeros((target_slices, 256, 256))
        cor_array = np.zeros((target_slices, 256, 256))
        sag_array = np.zeros((target_slices, 256, 256))
        mic_array = np.zeros((1, 256, 256))

        c_date = str(pd.to_datetime(collection_date)).split(" ")[0]
        base_date = datetime.strptime(c_date, '%Y-%m-%d').date()
        target_date_str = c_date.replace('-', '')


        with h5py.File("dataset/img_dataset.h5", "r") as f:
            pt_id_str = str(pt_id)
            patient_group = f[pt_id_str]
            date_to_use = None

            if target_date_str in patient_group.keys():
                date_to_use = target_date_str
            else:
                days_list = []
                for i in range(-14, 8):
                    target_date = base_date + timedelta(days=i)
                    days_list.append(target_date.strftime('%Y%m%d'))

                for day in days_list:
                    if day in patient_group.keys():
                        date_to_use = day
                        break

            if date_to_use is None:
                return {
                'axi_img': axi_array,
                'cor_img': cor_array,
                'sag_img': sag_array,
                'mic_img': mic_array,
        }


            output = patient_group[date_to_use]

            def load_image_sequence(img_group, target_array):
                if img_group is None:
                    return

                img_keys = sorted(img_group.keys())
                num_images = len(img_keys)

                if num_images == 0:
                    return

                load_count = min(num_images, target_slices)

                for idx in range(load_count):
                    img_data = img_group[img_keys[idx]][:]
                    target_array[idx, :, :] = img_data

            if "axi_img" in output.keys():
                load_image_sequence(output["axi_img"], axi_array)

            if "cor_img" in output.keys():
                load_image_sequence(output["cor_img"], cor_array)

            if "sag_img" in output.keys():
                load_image_sequence(output["sag_img"], sag_array)

            if "mic_img" in output.keys():
                mic_group = output["mic_img"]
                if len(mic_group.keys()) > 0:
                    first_key = sorted(mic_group.keys())[0]
                    mic_array[0, :, :] = mic_group[first_key][:]

        return {
                'axi_img': axi_array,
                'cor_img': cor_array,
                'sag_img': sag_array,
                'mic_img': mic_array
        }



    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):

        # return data
        return {
            "x_input": self.x_input[idx],
            "y_duration": self.y_duration[idx],
            "y_event": self.y_event[idx],
            "y_prog_risk": self.y_prog_risk[idx],
            "axi_img": self.axi_img[idx],
            "cor_img": self.cor_img[idx],
            "sag_img": self.sag_img[idx],
            "mic_img": self.mic_img[idx],
            "patient_id": self.pt_id[idx],
        }