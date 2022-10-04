import os.path as osp

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'
_FIELDS = 'extra_fields'

# Available datasets
COMMON_DATASETS = {
    'CIHP_train': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/CIHP/Training/Category_ids',
             'label_shift': 0},
    },
    'CIHP_val': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/CIHP/Validation/Category_ids',
             'label_shift': 0},
    },
    'CIHP_test': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_test.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             # 'seg_root': _DATA_DIR + '/CIHP/Testing/Category_ids',  # no gt seg
             'label_shift': 0},
    },
    'LIP_train': {
        _IM_DIR:
            _DATA_DIR + '/LIP/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LIP/Training/Category_ids',
             'label_shift': 0}
    },
    'LIP_val': {
        _IM_DIR:
            _DATA_DIR + '/LIP/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LIP/Validation/Category_ids',
             'label_shift': 0}
    },
    'LIP_test': {
        _IM_DIR:
            _DATA_DIR + '/LIP/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'label_shift': 0}
    },
    'ATR_train': {
        _IM_DIR:
            _DATA_DIR + '/ATR/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([9, 10], [12, 13], [14, 15]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/ATR/Training/Category_ids',
             'label_shift': 0}
    },
    'ATR_val': {
        _IM_DIR:
            _DATA_DIR + '/ATR/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([9, 10], [12, 13], [14, 15]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/ATR/Validation/Category_ids',
             'label_shift': 0}
    },
    'VIP_Fine_train': {
        _IM_DIR:
            _DATA_DIR + '/VIP/VIP_Fine/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/VIP/VIP_Fine/annotations/VIP_Fine_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/VIP/VIP_Fine/Training/Category_ids',
             'label_shift': 0}
    },
    'VIP_Fine_val': {
        _IM_DIR:
            _DATA_DIR + '/VIP/VIP_Fine/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/VIP/VIP_Fine/annotations/VIP_Fine_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/VIP/VIP_Fine/Validation/Category_ids',
             'label_shift': 0}
    },
    'VIP_Fine_test': {
        _IM_DIR:
            _DATA_DIR + '/VIP/VIP_Fine/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/VIP/VIP_Fine/annotations/VIP_Fine_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             # 'seg_root': _DATA_DIR + '/VIP/VIP_Fine/Testing/Category_ids',  # no gt seg
             'label_shift': 0}
    },
    'MHP-v2_train': {
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'parsing'],
             'seg_root': _DATA_DIR + '/MHP-v2/Training/Category_ids',
             'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]),
             'label_shift': 0},
    },
    'MHP-v2_val': {
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'parsing'],
             'seg_root': _DATA_DIR + '/MHP-v2/Validation/Category_ids',
             'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]),
             'label_shift': 0},
    },
    'MHP-v2_test': {
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'parsing'],
             # 'seg_root': _DATA_DIR + '/MHP-v2/Testing/Category_ids',  # no gt seg
             'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]),
             'label_shift': 0},
    },
    'PASCAL-Person-Part_train': {
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/PASCAL-Person-Part_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/PASCAL-Person-Part/Training/Category_ids',
             'label_shift': 0}
    },
    'PASCAL-Person-Part_test': {
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/PASCAL-Person-Part_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/PASCAL-Person-Part/Testing/Category_ids',
             'label_shift': 0}
    },
}
