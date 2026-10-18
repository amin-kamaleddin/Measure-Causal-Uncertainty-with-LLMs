"""Verified SHARE release 9.0.0 variable maps; update only after metadata audit."""
MAP = {
    (7, 'cc'): ['mergeid', 'country', 'cc733_', 'cc008_', 'cc729_'],
    (7, 'gv_weights'): ['mergeid', 'SHARELIFE'],
    (7, 'gv_health'): ['mergeid', 'eurod'],
    (7, 'cv_r'): ['mergeid', 'interview', 'int_year', 'int_month'],
    (8, 'ac'): ['mergeid', 'country', 'ac035d5', 'ac035d1', 'ac036_5'],
    (8, 'gv_health'): ['mergeid', 'eurod'],
    (8, 'cv_r'): ['mergeid', 'country', 'interview', 'age_int', 'gender', 'int_year', 'int_month', 'deceased'],
    (8, 'gv_isced'): ['mergeid', 'isced1997_r'],
    (8, 'gv_weights'): ['mergeid', 'cciw_w8_main', 'dw_w8', 'psu', 'stratum1'],
    (9, 'gv_health'): ['mergeid', 'eurod', 'eurodcat'],
    (9, 'cv_r'): ['mergeid', 'country', 'interview', 'int_year', 'int_month', 'deceased'],
}

LABELS = {'ac035d5': 'Gone to a sport, social or other kind of club in last year',
          'cc733_': 'Family financially: 1 well off; 2 average; 3 poor; 4 varied; 5 outside family',
          'eurod': 'Depression score: 0–12; high more depressive symptoms'}
