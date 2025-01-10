import os
import numpy as np
import nibabel as nib
from nilearn import masking
from nilearn.connectome import ConnectivityMeasure

# 设置路径
nifti_folder = r'G:\llb\rest\jx' # 患者NIfTI文件所在文件夹路径
roi_mask_path = r'G:\llb\2_Resliced3\Resliced3_basefor.nii' # ROI掩膜路径

# 获取文件夹中所有.nii文件
nifti_files = sorted([os.path.join(nifti_folder, f) for f in os.listdir(nifti_folder) if f.endswith('.nii')])

# 加载ROI掩膜
roi_mask_img = nib.load(roi_mask_path)
roi_mask_data = roi_mask_img.get_fdata().astype(bool)

# 初始化相关性计算
correlation_measure = ConnectivityMeasure(kind='correlation')

degree_centralities = []

for file in nifti_files:
    # 加载患者fMRI数据
    fmri_img = nib.load(file)
    fmri_data = fmri_img.get_fdata()

    # 提取ROI时间序列
    roi_time_series = masking.apply_mask(fmri_img, roi_mask_path)

    # 计算相关矩阵
    correlation_matrix = correlation_measure.fit_transform([roi_time_series])[0]

    # 计算度中心性
    degree_centrality = np.sum(correlation_matrix > 0.25, axis=1)  # 统计连接数
    degree_centralities.append(degree_centrality)


scm_dc = []
# 输出结果并计算ROI内的总度中心性
for i, centrality in enumerate(degree_centralities):
    sum_centrality = np.sum(centrality)  # 对ROI内的度中心性求和
    scm_dc.append(sum_centrality)  # 正确的调用append方法
    print(f'Patient {i + 1} Degree Centrality: {centrality}')
    print(f'Patient {i + 1} Total Degree Centrality (Sum of ROI): {sum_centrality}')


