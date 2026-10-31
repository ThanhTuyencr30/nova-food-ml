import pandas as pd
from src.nova.data.prepare import prepare_dataframe, add_vietnamese_flag, split_vietnam


def test_prepare_dataframe_filters_invalid_rows():
    df = pd.DataFrame({
        "ingredients_text": ["milk, salt", "", "sugar"],
        "nova_group": [1, 2, 5],
    })
    result = prepare_dataframe(df)
    assert len(result) == 1
    assert result.iloc[0]["nova_group"] == 1


def test_split_vietnam_separates_rows():
    df = pd.DataFrame({
        "ingredients_text": ["đường, muối", "sugar, salt"],
        "nova_group": [2, 2],
        "is_vietnam": [1, 0],
    })
    others, vietnam = split_vietnam(add_vietnamese_flag(df))
    assert len(others) == 1 and len(vietnam) == 1
    assert vietnam.iloc[0]["co_chu_viet"] == 1
    assert others.iloc[0]["co_chu_viet"] == 0