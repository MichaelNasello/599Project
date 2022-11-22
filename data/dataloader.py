import os.path

import pandas as pd


def load_patient_task_data_from_txt(patient_id, task_num):
    assert patient_id in ['001', '002', '003', '004', '005', '006', '007', '008-1', '008-2', '009', '010', '011', '012']
    assert 1 <= task_num <= 6

    # Define column headers
    eeg_columns = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'FC1',
        'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
    ]
    imu_columns = []
    for i, body_part in enumerate(['left_shank', 'right_shank', 'waist', 'arm']):
        imu_columns.append(f'acc_x_{body_part}')
        imu_columns.append(f'acc_y_{body_part}')
        imu_columns.append(f'acc_z_{body_part}')
        imu_columns.append(f'gyro_x_{body_part}')
        imu_columns.append(f'gyro_y_{body_part}')
        imu_columns.append(f'gyro_z_{body_part}')

        if i < 3:
            imu_columns.append(f'NC_invalid_{i}')
        elif i == 3:
            imu_columns.append('SC')

    # EMG, ECG, EOG values vary based on patient and require manual effort
    emg_ecg_eog_columns = []

    # EMG1,2
    if patient_id in ['001', '002', '006', '007', '008-1', '008-2']:
        emg_ecg_eog_columns.append('RTA')
        if patient_id != '008-2':
            emg_ecg_eog_columns.append('LTA')
        else:
            emg_ecg_eog_columns.append('RGS')
    else:
        emg_ecg_eog_columns.append('LTA')
        emg_ecg_eog_columns.append('RTA')
    emg_ecg_eog_columns.append('IO')

    # EMG3
    if patient_id != '009':
        emg_ecg_eog_columns.append('ECG')
    else:
        emg_ecg_eog_columns.append('RGS')

    # EMG4
    if patient_id == '008-2':
        emg_ecg_eog_columns.append('LTA')
    elif patient_id == '009':
        emg_ecg_eog_columns.append('ECG')
    else:
        emg_ecg_eog_columns.append('RGS')

    headers = ['time'] + eeg_columns + emg_ecg_eog_columns + imu_columns + ['label']

    # Read data
    if patient_id in ['008-1', '008-2']:
        patient_folder = f'{patient_id.split("-")[0]}/OFF_{patient_id.split("-")[1]}'
    else:
        patient_folder = patient_id

    if os.path.exists(f'raw_data/{patient_folder}/task_{task_num}.txt'):
        data = pd.read_table(f'raw_data/{patient_folder}/task_{task_num}.txt', sep=',')
    else:
        return pd.DataFrame()

    data = data.drop('0', axis=1)
    data = data.set_axis(headers, axis=1)

    return data


def clean_and_verify(raw_data):
    if raw_data.empty:
        return raw_data

    eeg_columns = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'FC1',
        'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
    ]

    for i in range(3):
        raw_data = raw_data.drop(f'NC_invalid_{i}', axis=1)

    # Right and left shank not both included
    for axis in ['x', 'y', 'z']:
        for sensor in ['acc', 'gyro']:
            col = f'{sensor}_{axis}_left_shank'
            if raw_data[col].values.mean() != 0 and raw_data[col].values.std() != 0:
                raw_data[f'{sensor}_{axis}_shank'] = raw_data[col].values
                raw_data = raw_data.drop([col, f'{sensor}_{axis}_right_shank'], axis=1)
            else:
                raw_data[f'{sensor}_{axis}_shank'] = raw_data[f'{sensor}_{axis}_right_shank'].values
                raw_data = raw_data.drop([col, f'{sensor}_{axis}_right_shank'], axis=1)

    for col in raw_data.columns[1:]:
        # Not using EEG data
        if col in eeg_columns:
            raw_data = raw_data.drop(col, axis=1)
            continue

        if raw_data[col].values.mean() == 0 and raw_data[col].values.std() == 0:
            # print(f'Dropping {col} (has no useful information)')

            raw_data = raw_data.drop(col, axis=1)

    return raw_data


if __name__ == '__main__':
    patient_data = load_patient_task_data_from_txt(patient_id='001', task_num=1)
    clean_patient_data = clean_and_verify(patient_data)
    print(clean_patient_data.head())
