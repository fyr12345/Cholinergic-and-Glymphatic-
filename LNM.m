clc
clear

% Define ROI directory and NIfTI files directory
roi_dir = 'H:/llb/lesion_Resliced'; % Update this path to your ROI directory
nifti_files_dir = 'H:/JIYA/1000';
output_dir = 'H:/llb/1/roifc_ttest_results';
% Create output directory (if it does not exist)
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Get list of NIfTI files
nifti_files = dir(fullfile(nifti_files_dir, '*.nii'));

% Get list of ROI files
roi_files = dir(fullfile(roi_dir, '*.nii'));

% Specify the number of cores (workers) you want to use
num_cores = 20; % Change this number to the desired number of cores

% Start a parallel pool with the specified number of workers
if isempty(gcp('nocreate'))
    parpool('local', num_cores);
end

% Load a sample NIfTI file to get dimensions
sample_nifti_file = fullfile(nifti_files(1).folder, nifti_files(1).name);
sample_nifti_img = spm_vol(sample_nifti_file);
sample_nifti_data = spm_read_vols(sample_nifti_img);
[x_dim, y_dim, z_dim, num_timepoints] = size(sample_nifti_data);
num_voxels = x_dim * y_dim * z_dim;

% Preallocate a cell array to store outputs (since parfor cannot write to files directly)
t_maps_cell = cell(length(roi_files), 1);
roi_names = cell(length(roi_files), 1);

% For each ROI file, use parfor to parallelize
parfor r = 1:length(roi_files)
    roi_file = fullfile(roi_files(r).folder, roi_files(r).name);
    
    % Load ROI data
    roi_img = spm_vol(roi_file);
    roi_data = spm_read_vols(roi_img);
    roi_indices = find(roi_data > 0);
    
    % Initialize connectivity maps matrix: (num_subjects x num_voxels)
    num_subjects = length(nifti_files);
    connectivity_maps = zeros(num_subjects, num_voxels);
    
    % For each NIfTI file (subject)
    for i = 1:num_subjects
        nifti_file = fullfile(nifti_files(i).folder, nifti_files(i).name);
        nifti_img = spm_vol(nifti_file);
        nifti_data = spm_read_vols(nifti_img);
    
        % Ensure ROI and NIfTI data dimensions match
        if any(size(roi_data) ~= size(nifti_data(:, :, :, 1)))
            error('ROI dimensions do not match NIfTI file dimensions.');
        end
    
        % Extract ROI time series
        num_timepoints = size(nifti_data, 4);
        roi_time_series = zeros(num_timepoints, 1);
        
        for t = 1:num_timepoints
            volume = nifti_data(:, :, :, t);
            roi_time_series(t) = mean(volume(roi_indices));
        end
    
        % Extract whole brain time series
        brain_time_series = reshape(nifti_data, [], num_timepoints)';
    
        % Compute functional connectivity (correlation)
        correlation_vector = corr(roi_time_series, brain_time_series);
    
        % Fisher Z transform
        z_vector = atanh(correlation_vector);
    
        % Store the z_vector in the connectivity_maps matrix
        connectivity_maps(i, :) = z_vector;
    end
    
    % Perform voxel-wise one-sample t-test across subjects
    [~, ~, ~, stats] = ttest(connectivity_maps, 0, 'Alpha', 0.05, 'Dim', 1);
    t_values = stats.tstat;
    
    % Reshape t_values into brain dimensions
    t_map = reshape(t_values, x_dim, y_dim, z_dim);
    
    % Store the t_map and ROI name in the cell arrays
    t_maps_cell{r} = t_map;
    [~, roi_name, ~] = fileparts(roi_files(r).name);
    roi_names{r} = roi_name;
end

% After the parfor loop, save the t_maps to NIfTI files
for r = 1:length(roi_files)
    roi_file = fullfile(roi_files(r).folder, roi_files(r).name);
    roi_img = spm_vol(roi_file);
    output_nii = roi_img; % Use ROI image's metadata
    output_nii.dt = [16, 0]; % Data type as float
    output_nii.fname = fullfile(output_dir, [roi_names{r}, '_ttest.nii']);
    spm_write_vol(output_nii, t_maps_cell{r});
end

% Shut down the parallel pool (optional)
delete(gcp('nocreate'));