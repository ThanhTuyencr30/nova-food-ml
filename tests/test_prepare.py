import pandas as pd
from src.nova.data.prepare import prepare_dataframe

def test_prepare_dataframe_filters_invalid_rows():
    df = pd.DataFrame({
        "ingredients_text": ["milk, salt", "", "sugar"],
        "nova_group": [1, 2, 5],
    })
    result = prepare_dataframe(df)
    assert len(result) == 1
    assert result.iloc[0]["nova_group"] == 1
