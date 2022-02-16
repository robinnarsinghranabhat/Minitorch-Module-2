'''
Additional Units tests 
Author : robinnarsingha123@gmail.com
'''

import minitorch
import pytest

@pytest.mark.task2_1
def test_to_index():
    
    ordinal, shape, out_index = 5 , (4,2) , [0,0]
    minitorch.to_index(ordinal, shape, out_index)
    assert out_index == [2,1]