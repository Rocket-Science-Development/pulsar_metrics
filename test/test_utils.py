from ..pulsar_metrics.utils import *
from ..pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg

# Testing dataframe validation
#==============================

def test_dataframe_is_valid():
  df = pd.DataFrame(np.nan, columns = ['y_pred', 'y_true', 'date', 'model_id'] )
  assert validate_dataframe(data = df) == True
  
def test_validate_target_is_missing():
  df = pd.DataFrame(np.nan, columns = ['y_pred', 'date', 'model_id'] )
  with pytest.raises(error_msg, match = ERROR_MSG_MISSING_KEY):
    validate_dataframe(data = df)
  
def test_validate_prediction_is_missing():
  df = pd.DataFrame(np.nan, columns = ['y_true', 'date', 'model_id'] )
  with pytest.raises(error_msg, match = ERROR_MSG_MISSING_KEY):
    validate_dataframe(data = df)
  
def test_validate_date_is_missing():
  df = pd.DataFrame(np.nan, columns = ['y_true', 'y_pred', 'model_id'] )
  with pytest.raises(error_msg, match = ERROR_MSG_MISSING_KEY):
    validate_dataframe(data = df)
  
def test_validate_modelid_is_missing():
  df = pd.DataFrame(np.nan, columns = ['y_true', 'y_pred', 'date'] )
  with pytest.raises(error_msg, match = ERROR_MSG_MISSING_KEY):
    validate_dataframe(data = df)

# Testing comparison to threshold
#===================================

def test_compare_to_single_threshold():
  value = 3
  threshold = 2
  assert compare_to_threshold(value, threshold) == False
  
def test_compare_to_interval_threshold():
  value = 3
  threshold = [1, 4]
  assert compare_to_threshold(value, threshold) == True
  
def test_invalid_threshold_with_three_elements():
  value = 3
  threshold = [1, 4, 6]
  with pytest.raises(ValueError, match=ERROR_MSG_VECTOR_THRESHOLD):
    compare_to_threshold(value, threshold)
    
def test_invalid_threshold_type():
  value = 3
  threshold = '1'
  with pytest.raises(ValueError, match=ERROR_MSG_VECTOR_THRESHOLD):
    compare_to_threshold(value, threshold)
  
 
