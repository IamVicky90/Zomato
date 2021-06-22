import os
def test_pass():
    lst=os.listdir('models')
    if len(lst)==4:
        print('Program is Ready For Prediction')
        assert True
    else:
        print('Program is not Ready For Prediction')
        assert False