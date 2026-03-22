from models.trainer import Trainer

slice_num = 5
batch_size = 32
lr = 0.0005
epochs = 50
ablation = "t2"   # ["Full", "no_time", "no_img", "no_mic", "no_mri", "no_pret","modal_full","t2"]
pre_train_resnet_path = "../TEMPO/dataset/MRI_pretrained.pth"
tab_df_path = "../TEMPO/dataset/tabular_input.csv"



if __name__ == "__main__":
    tempo = Trainer(slice_num, batch_size, lr, epochs, ablation, pre_train_resnet_path, tab_df_path)
    tempo.train_pipeline()