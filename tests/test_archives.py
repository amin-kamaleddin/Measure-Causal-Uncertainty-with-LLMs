import pytest
from sharecausal.data.archives import locate_archives,read_module,metadata,archive_inventory

def test_read_selectively(mock_archives):
    paths=locate_archives(mock_archives)
    d=read_module(paths[8],8,'ac',['mergeid','ac035d5'])
    assert d.shape==(40,2)
    assert int((d.ac035d5==-2).sum())==1
    md=metadata(paths[7],7,'cc');assert md['rows']==40
    assert 'cc733_' in md['columns']
    assert archive_inventory(paths)['8']['stata_modules']>=5

def test_missing_module_fails(mock_archives):
    with pytest.raises(KeyError):
        read_module(locate_archives(mock_archives)[7],7,'does_not_exist')

def test_missing_column_fails(mock_archives):
    with pytest.raises(ValueError,match='missing required columns'):
        read_module(locate_archives(mock_archives)[8],8,'ac',['mergeid','NONEXISTENT'])

def test_missing_zip_raises(tmp_path):
    with pytest.raises(FileNotFoundError):locate_archives(tmp_path)
