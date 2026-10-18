from io import BytesIO
import pytest
from adaptive_defense.uploads import read_upload,MAX_BYTES,MAX_ROWS

def test_valid_upload():
    assert len(read_upload(BytesIO(b"dur,proto\n1,tcp\n")))==1

@pytest.mark.parametrize("content",[b"",b"dur\n",b"\xff\xfe"])
def test_invalid_upload(content):
    with pytest.raises(ValueError):
        read_upload(BytesIO(content))

def test_oversized_upload():
    with pytest.raises(ValueError,match="10 MB"):
        read_upload(BytesIO(b"x"*(MAX_BYTES+1)))

def test_too_many_rows():
    with pytest.raises(ValueError,match="10,000"):
        read_upload(BytesIO(b"dur\n"+b"1\n"*(MAX_ROWS+1)))
