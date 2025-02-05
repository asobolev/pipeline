# processed sessions with LFP channel numbers

selected_009266 = {
    # not fully processed
    #'009266_hippoSIT_2023-04-13_08-57-46': {'A1': 17, 'PPC': 32},  # ch17, very little AEPs
    #'009266_hippoSIT_2023-04-14_09-17-34': {'A1': 17, 'PPC': 32},
    #'009266_hippoSIT_2023-04-17_09-06-10': {'A1': 17, 'PPC': 32},  # little AEPs, 7 + 55 correction, 5463 events, frequency

    # A1 - PPC
    '009266_hippoSIT_2023-04-17_17-04-17': {'A1': 17, 'PPC': 32},  # 20 + 55 correction, 5067 events
    '009266_hippoSIT_2023-04-18_10-10-37': {'A1': 17, 'PPC': 32},  # 10 + 55 correction, 5682 events
    '009266_hippoSIT_2023-04-18_17-03-10': {'A1': 17, 'PPC': 32},  # 7 + 55 correction, 5494 events
    '009266_hippoSIT_2023-04-19_10-33-51': {'A1': 17, 'PPC': 32},  # 4 + 55 correction, 6424 events
    '009266_hippoSIT_2023-04-20_08-57-39': {'A1': 1, 'PPC': 32},   # 1 + 55 correction, 6424 events
    #'009266_hippoSIT_2023-04-20_15-24-14': {'A1': 20, 'PPC': 32},  # 2 + 55 correction, 5612 events
    '009266_hippoSIT_2023-04-21_08-43-00': {'A1': 20, 'PPC': 32},  # 14 + 55 correction, 6282 events
    '009266_hippoSIT_2023-04-21_13-12-31': {'A1': 20, 'PPC': 32},  # 15 + 55 correction, 6041 events
    '009266_hippoSIT_2023-04-24_10-08-11': {'A1': 20, 'PPC': 40},  # 21 + 55 correction, 5579 events
    '009266_hippoSIT_2023-04-24_16-56-55': {'A1': 17, 'PPC': 32},  # 5 + 55* correction, 6165 events, frequency
    '009266_hippoSIT_2023-04-26_08-20-17': {'A1': 17, 'PPC': 32},  # 12 + 55* correction, 6095 events, duration
    '009266_hippoSIT_2023-05-02_12-22-14': {'A1': 20, 'PPC': 40},  # 10 + 55 correction, 5976 events, frequency
    #'009266_hippoSIT_2023-05-04_09-11-06': {'A1': 17, 'PPC': 32},  # 5 + 55* correction, 4487 events, coma session
    '009266_hippoSIT_2023-05-04_19-47-15': {'A1': 20, 'PPC': 32},  # 2 + 55 correction, 5678 events, frequency

    # A1 - HPC
    '009266_hippoSIT_2023-05-22_09-27-22': {'A1': 9, 'HPC': 56},
    '009266_hippoSIT_2023-05-23_09-18-05': {'A1': 9, 'HPC': 56},
    '009266_hippoSIT_2023-05-25_15-55-57': {'A1': 9, 'HPC': 56},
    '009266_hippoSIT_2023-06-14_08-21-23': {'A1': 9, 'HPC': 56},
    '009266_hippoSIT_2023-06-19_08-58-35': {'A1': 9, 'HPC': 56},
}

# Old PPC sessions (P1 is on 30ms)
selected_008229 = {
    '008229_hippoSIT_2022-05-16_20-36-44': {"AEPs": {'PPC': 0}, "ephys": {"offset": 98}}, # chs: 0, 56; 91 corr
    '008229_hippoSIT_2022-05-17_21-44-43': {"AEPs": {'PPC': 0}, "ephys": {"offset": 97}}, # chs: 0, 31, 54, 56; 103 corr
    '008229_hippoSIT_2022-05-18_14-36-18': {"AEPs": {'PPC': 0}, "ephys": {"offset": 105}}, # chs: 0, 56; 70 corr
    '008229_hippoSIT_2022-05-20_15-54-39': {"AEPs": {'PPC': 0}, "ephys": {"offset": 86}}, # chs: 0, 56; 65 corr
}


selected_009265 = {
    '009265_hippoSIT_2023-02-24_09-53-26': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 116}},  # 90 corr
    '009265_hippoSIT_2023-02-24_17-22-46': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 134}},  # 108 corr
    '009265_hippoSIT_2023-02-27_10-18-32': {"AEPs": {'A1': 16, 'PPC': 32}, "ephys": {"offset": 97}},  # 66 corr
    '009265_hippoSIT_2023-02-27_15-33-46': {"AEPs": {'A1': 29, 'PPC': 32}, "ephys": {"offset": 107}},  # 105 corr
    '009265_hippoSIT_2023-02-28_09-16-50': {"AEPs": {'A1': 29, 'PPC': 32}, "ephys": {"offset": 101}},  # 100 corr
    '009265_hippoSIT_2023-02-28_13-16-10': {"AEPs": {'A1': 48, 'PPC': 32}, "ephys": {"offset": 104}},  # 106 corr
    '009265_hippoSIT_2023-02-28_20-45-04': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 103}},  # 87 corr
    '009265_hippoSIT_2023-03-01_10-46-12': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 101}},  # 83 corr
    '009265_hippoSIT_2023-03-02_09-32-54': {"AEPs": {'A1': 28, 'PPC': 40}, "ephys": {"offset": 105}},  # 101 corr
    '009265_hippoSIT_2023-03-02_16-27-42': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 124}},  # 114 corr
    '009265_hippoSIT_2023-03-02_20-11-35': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 105}},  # 83 corr
    '009265_hippoSIT_2023-03-03_09-37-07': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 95}},  # 85 corr
    '009265_hippoSIT_2023-03-03_16-00-47': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 111}},  # 102 corr
    '009265_hippoSIT_2023-03-04_11-12-04': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 93}},  # 94 corr
    '009265_hippoSIT_2023-03-05_11-52-17': {"AEPs": {'A1': 48, 'PPC': 0}, "ephys": {"offset": 108}},   # 66 corr
    '009265_hippoSIT_2023-03-05_18-31-32': {"AEPs": {'A1': 48, 'PPC': 35}, "ephys": {"offset": 97}},  # 43 corr
    '009265_hippoSIT_2023-03-08_18-10-07': {"AEPs": {'A1': 48, 'PPC': 37}, "ephys": {"offset": 100}},  # 102 corr
    '009265_hippoSIT_2023-03-09_20-03-08': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 97}},  # 100 corr
    '009265_hippoSIT_2023-03-10_09-57-34': {"AEPs": {'A1': 48, 'PPC': 40}, "ephys": {"offset": 96}},  # 98 corr
    '009265_hippoSIT_2023-04-13_09-54-39': {"AEPs": {'A1': 1, 'PPC': 33}, "ephys": {"offset": 5}},   # 258 corr
    '009265_hippoSIT_2023-04-20_11-39-02': {"AEPs": {'A1': 16, 'PPC': 33}, "ephys": {"offset": 0}},  # 0 corr??
}


selected_60 = {
    '60_SIT_2023-11-24_17-09-10': {'A1': 16, 'PPC': 55},  # 0 corr??
}


selected_57 = [
    "57_SIT_2023-12-18_14-07-34",
    "57_SIT_2023-12-22_14-08-07",
    "57_SIT_2023-12-22_14-43-58",
    "57_SIT_2023-12-22_17-37-18",
    "57_SIT_2023-12-23_14-21-01",
    "57_SIT_2023-12-28_16-43-28",
    "57_SIT_2023-12-29_11-06-26",
    "57_SIT_2023-12-29_11-40-14",
    "57_SIT_2023-12-29_12-11-46",
    "57_SIT_2024-01-02_14-43-18",
    "57_SIT_2024-01-02_16-38-05",
    "57_SIT_2024-01-02_17-10-09",
    "57_SIT_2024-01-03_19-22-18",
    "57_SIT_2024-01-03_19-54-59",
    "57_SIT_2024-01-04_14-16-22",
    "57_SIT_2024-01-04_14-52-59",
    "57_SIT_2024-01-05_14-35-49",
    "57_SIT_2024-01-05_15-08-34",
    "57_SIT_2024-01-06_16-52-40",
    "57_SIT_2024-01-06_17-25-35",
    "57_SIT_2024-01-07_19-23-28",
    "57_SIT_2024-01-08_15-51-26",
    "57_SIT_2024-01-12_13-23-02",
    "57_SIT_2024-01-15_13-45-22",
    "57_SIT_2024-01-15_14-34-48"
]