# pylint: disable=too-many-lines
import datetime
from functools import reduce
from typing import Any, NamedTuple, Optional

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Column, DataFrame
from pyspark.sql.window import Window, WindowSpec

from constants import FIRST_FLIGHT_DATE_A400M


def filter_by_column_values(
    df: DataFrame, col_names_filters: dict[str, list]
) -> DataFrame:
    """
    Filters a data frame by the column names and given values defined as in a dictionary.
    If `exclude_values=True` is set, the filtering is negated, i.e. the given values
    are removed.

    >>> spark = getfixture('spark')
    >>> sar_loop_values = spark.createDataFrame([
    ...     Row(alpha='ALPHA_1', ac_tail_no='F-AAA', year=2022, value_float=1),
    ...     Row(alpha='ALPHA_1', ac_tail_no='F-AAA', year=2022, value_float=1),
    ...     Row(alpha='ALPHA_1', ac_tail_no='F-AAA', year=2020, value_float=1),
    ...     Row(alpha='ALPHA_1', ac_tail_no='F-AAB', year=2021, value_float=1),
    ...     Row(alpha='ALPHA_2', ac_tail_no='F-AAC', year=2022, value_float=1),
    ...     Row(alpha='ALPHA_2', ac_tail_no='F-AAC', year=2021, value_float=1),
    ...     Row(alpha='ALPHA_2', ac_tail_no='F-AAC', year=2020, value_float=1),
    ...     Row(alpha='ALPHA_2', ac_tail_no='F-AAA', year=2021, value_float=1),
    ...     Row(alpha='XXXXXXX', ac_tail_no='F-AAA', year=2021, value_float=1),
    ...     ])
    >>> col_names_filters = {'alpha': ['ALPHA_1'], 'ac_tail_no': ['F-AAA']}
    >>> filter_by_column_values(sar_loop_values, col_names_filters).show()
    +-------+----------+----+-----------+
    |  alpha|ac_tail_no|year|value_float|
    +-------+----------+----+-----------+
    |ALPHA_1|     F-AAA|2022|          1|
    |ALPHA_1|     F-AAA|2022|          1|
    |ALPHA_1|     F-AAA|2020|          1|
    +-------+----------+----+-----------+
    <BLANKLINE>
    """
    for col_name, values in col_names_filters.items():
        df = df.where(F.col(col_name).isin(values))
    return df


def filter_single_column_by_values(
    df: DataFrame,
    col_name: str,
    values: list[FilterValue],
    chain_conditions: str = "OR",
) -> DataFrame:
    """
    Filters a single column by its values. The value_filters arguments is a list of tuples, where each of the tuples
    contains the information for filtering by a specific value:

    FilterValue(value=..., fuzzy=..., exclude=...)
        - `value` is the value to filter by
        - `fuzzy` is a boolean flag that allows for more fuzzy filtering (currently: match substrings)
        - `exclude` negates the filtering condition, i.e. allows to exclude a value

    `chain_conditions` allows to pick between logical OR/AND when joining multiple conditions.

    Similar to filter_by_column_values, but allows for a more refined filtering.
    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(value='ALPHA'),
    ...     Row(value='ALPHA BRAVO'),
    ...     Row(value='BRAVO CHARLIE'),
    ...     Row(value='CHARLIE'),
    ... ])
    >>> filter_single_column_by_values(df, 'value', [FilterValue(value='ALPHA')]).show(truncate=False)
    +-----+
    |value|
    +-----+
    |ALPHA|
    +-----+
    <BLANKLINE>
    >>> filter_single_column_by_values(df, 'value', [FilterValue(value='BRAVO', fuzzy=True)]).show(truncate=False)
    +-------------+
    |value        |
    +-------------+
    |ALPHA BRAVO  |
    |BRAVO CHARLIE|
    +-------------+
    <BLANKLINE>
    >>> value_filters = [FilterValue(value='BRAVO', fuzzy=True, exclude=True), FilterValue(value='CHARLIE', fuzzy=True, exclude=True)]
    >>> filter_single_column_by_values(df, 'value', value_filters, chain_conditions = 'AND').show(truncate=False)
    +-----+
    |value|
    +-----+
    |ALPHA|
    +-----+
    <BLANKLINE>
    """
    value_condition_list = []
    for filter_value in values:
        if filter_value.fuzzy:
            condition = F.col(col_name).contains(filter_value.value)
        else:
            condition = F.col(col_name) == filter_value.value
        if filter_value.exclude:
            condition = ~condition
        value_condition_list.append(condition)

    if chain_conditions == "AND":
        filter_condition = reduce(lambda x, y: x & y, value_condition_list)
    elif chain_conditions == "OR":
        filter_condition = reduce(lambda x, y: x | y, value_condition_list)
    else:
        raise ValueError('chain_condition should be one of the following: "OR", "AND".')

    return df.where(filter_condition)


def rename_columns(df: DataFrame, column_names: dict[str, str]) -> DataFrame:
    """
    This method renames column(s).

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(col_1='Foo', col_2='Bar')
    ... ])

    >>> column_names = {'col_1':'new_col_1', 'col_2':'new_col_2'}
    >>> rename_columns(df, column_names).show()
    +---------+---------+
    |new_col_1|new_col_2|
    +---------+---------+
    |      Foo|      Bar|
    +---------+---------+
    <BLANKLINE>
    """
    for old_col_name, new_col_name in column_names.items():
        df = df.withColumnRenamed(old_col_name, new_col_name)
    return df


def rename_or_drop_columns(
    df: DataFrame, column_names: dict[str, Optional[str]]
) -> DataFrame:
    """
    Rename or drop columns.

    If a column is mentioned, its name is changed to the new name.
    If a column is mentioned and the new name is None, it will be dropped.
    If a column is not mentioned, it will be dropped.

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(col_1='Foo', col_2='Bar', col_3='Baz', col_4='Qux')
    ... ])

    >>> column_names = {'col_1':'new_col_1', 'col_2':'new_col_2', 'col_3': None}
    >>> rename_or_drop_columns(df, column_names).show()
    +---------+---------+
    |new_col_1|new_col_2|
    +---------+---------+
    |      Foo|      Bar|
    +---------+---------+
    <BLANKLINE>
    """
    return df.select(
        [F.col(old).alias(new) for old, new in column_names.items() if new]
    )


def create_constant_columns(df: DataFrame, column_names: dict[str, str]) -> DataFrame:
    """
    This methods creates new columns with a marker.

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(col_1='Foo'),
    ... ])

    >>> column_names = {'col_2':'marker_1', 'col_3':'marker_2'}
    >>> create_constant_columns(df, column_names).show()
    +-----+--------+--------+
    |col_1|   col_2|   col_3|
    +-----+--------+--------+
    |  Foo|marker_1|marker_2|
    +-----+--------+--------+
    <BLANKLINE>
    """
    for column_name, marker in column_names.items():
        df = df.withColumn(column_name, F.lit(marker))
    return df


def add_missing_columns_from_schema(df: DataFrame, schema: T.StructType) -> DataFrame:
    """
    This method adds columns based on a specified schema to a dataframe
    and then only selects the columns defined in the given schema.

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(col_1='foo', col_2=1),
    ... ])

    >>> schema = T.StructType(
    ... [
    ...     T.StructField('col_1', T.StringType()),
    ...     T.StructField('col_2', T.LongType()),
    ...     T.StructField('col_3', T.BooleanType()),
    ...     T.StructField('col_4', T.StringType()),
    ... ])

    >>> add_missing_columns_from_schema(df, schema).show()
    +-----+-----+-----+-----+
    |col_1|col_2|col_3|col_4|
    +-----+-----+-----+-----+
    |  foo|    1| null| null|
    +-----+-----+-----+-----+
    <BLANKLINE>

    >>> add_missing_columns_from_schema(df, schema).printSchema()
    root
     |-- col_1: string (nullable = true)
     |-- col_2: long (nullable = true)
     |-- col_3: boolean (nullable = true)
     |-- col_4: string (nullable = true)
    <BLANKLINE>
    """
    columns = [struct_field.name for struct_field in schema.fields]

    df = df.select(
        *df.columns,
        *[
            (F.lit(None).cast(struct_field.dataType)).alias(struct_field.name)
            for struct_field in schema.fields
            if struct_field.name not in df.columns
        ],
    )
    return df.select(*columns)


def create_id_column(
    df: DataFrame,
    id_column_name: str,
    column_list: list[str],
    sep: str = "|",
    apply_hash: bool = False,
) -> DataFrame:
    """
    This method creates a column containing an identifier
    which is created by concatenating a given column list.

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(col_1='foo', col_2='bar', col_3=None),
    ...     Row(col_1='foo', col_2='bar', col_3='foobar'),
    ... ])
    >>> id_column_name = 'new_ID'
    >>> column_list = ['col_1', 'col_2']
    >>> create_id_column(df, id_column_name, column_list, apply_hash=False).show()
    +-----+-----+------+-------+
    |col_1|col_2| col_3| new_ID|
    +-----+-----+------+-------+
    |  foo|  bar|  null|foo|bar|
    |  foo|  bar|foobar|foo|bar|
    +-----+-----+------+-------+
    <BLANKLINE>
    >>> create_id_column(df, id_column_name, column_list, apply_hash=True).show(truncate=False)
    +-----+-----+------+----------------------------------------------------------------+
    |col_1|col_2|col_3 |new_ID                                                          |
    +-----+-----+------+----------------------------------------------------------------+
    |foo  |bar  |null  |c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2|
    |foo  |bar  |foobar|c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2|
    +-----+-----+------+----------------------------------------------------------------+
    <BLANKLINE>

    >>> column_list_including_none_column = ['col_1', 'col_2', 'col_3']
    >>> create_id_column(df, id_column_name, column_list_including_none_column, apply_hash=False).show(truncate=False)
    +-----+-----+------+--------------+
    |col_1|col_2|col_3 |new_ID        |
    +-----+-----+------+--------------+
    |foo  |bar  |null  |foo|bar       |
    |foo  |bar  |foobar|foo|bar|foobar|
    +-----+-----+------+--------------+
    <BLANKLINE>
    >>> create_id_column(df, id_column_name, column_list_including_none_column, apply_hash=True).show(truncate=False)
    +-----+-----+------+----------------------------------------------------------------+
    |col_1|col_2|col_3 |new_ID                                                          |
    +-----+-----+------+----------------------------------------------------------------+
    |foo  |bar  |null  |c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2|
    |foo  |bar  |foobar|25cfe5b055cf6b1fd5205f36a43c9a0eb12d3b67a6064973d69368e186d19b62|
    +-----+-----+------+----------------------------------------------------------------+
    <BLANKLINE>
    """
    BIT_LENGTH = 256

    if apply_hash:
        sep = ""
        return df.withColumn(
            id_column_name, F.sha2(F.concat_ws(sep, *column_list), BIT_LENGTH)
        )

    return df.withColumn(id_column_name, F.concat_ws(sep, *column_list))


def replace_none_by_preceding_value(
    df: DataFrame, col_name: str, window: WindowSpec
) -> DataFrame:
    """
    >>> spark = getfixture('spark')
    >>> test_df = spark.createDataFrame([
    ...     Row(category = 1, timestamp=datetime.datetime(2020,1,1), value=1.),
    ...     Row(category = 1, timestamp=datetime.datetime(2020,1,2), value=3.),
    ...     Row(category = 1, timestamp=datetime.datetime(2020,1,3), value=None),
    ...     Row(category = 2, timestamp=datetime.datetime(2020,1,1), value=1.),
    ...     Row(category = 2, timestamp=datetime.datetime(2020,1,2), value=2.),
    ...     Row(category = 2, timestamp=datetime.datetime(2020,1,3), value=None),
    ... ])
    >>> test_window = Window.partitionBy('category').orderBy('timestamp')
    >>> replace_none_by_preceding_value(test_df, 'value', test_window).show()
    +--------+-------------------+-----+
    |category|          timestamp|value|
    +--------+-------------------+-----+
    |       1|2020-01-01 00:00:00|  1.0|
    |       1|2020-01-02 00:00:00|  3.0|
    |       1|2020-01-03 00:00:00|  3.0|
    |       2|2020-01-01 00:00:00|  1.0|
    |       2|2020-01-02 00:00:00|  2.0|
    |       2|2020-01-03 00:00:00|  2.0|
    +--------+-------------------+-----+
    <BLANKLINE>
    """
    return (
        df.withColumn(f"lagged_{col_name}", F.lag(col_name).over(window))
        .withColumn(col_name, F.coalesce(F.col(col_name), F.col(f"lagged_{col_name}")))
        .drop(f"lagged_{col_name}")
    )


def replace_column_value_by_regex(
    df: DataFrame, target_column: str, new_column: str, regex_list: list[str]
) -> DataFrame:
    r"""
    This method removes a specific pattern from a value in a given target column defined by
    regular expression and writes the string which is left into a new column.

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(col_1='C01_VALUE1'),
    ...     Row(col_1='C1_VALUE2'),
    ...     Row(col_1='FOO_VALUE2'),
    ... ])

    >>> regex_list = [r'(^C\d\d_)', r'(^C\d_)', r'(^FOO_)']
    >>> replace_column_value_by_regex(df, 'col_1', 'new_col', regex_list).show()
    +----------+-------+
    |     col_1|new_col|
    +----------+-------+
    |C01_VALUE1| VALUE1|
    | C1_VALUE2| VALUE2|
    +----------+-------+
    <BLANKLINE>
    """
    df = df.withColumn(new_column, F.col(target_column))
    for regex in regex_list:
        df = df.withColumn(new_column, F.regexp_replace(new_column, regex, ""))

    return df.dropDuplicates([new_column])


def replace_values(
    df: DataFrame,
    column_name: str,
    old_value_list: list[str],
    new_value_list: list[str],
):
    """
    Replaces the values from old_value_list in the column_name column by the values in the new_value_list

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(id=1, column='old_value_1'),
    ...     Row(id=2, column='old_value_2'),
    ...  ])
    >>> replace_values(df, 'column', ['old_value_1', 'old_value_2'], ['new_value_1', 'new_value_2']).show(truncate=False)
    +---+-----------+
    |id |column     |
    +---+-----------+
    |1  |new_value_1|
    |2  |new_value_2|
    +---+-----------+
    <BLANKLINE>
    """
    return df.replace(old_value_list, new_value_list, column_name)


def create_new_column_from_varying_columns(
    df: DataFrame, dependend_column: str, mapping_dict_dict: dict[str, dict[str, str]]
) -> DataFrame:
    """
    This function creates a new columns for each key in the mapping_dict_dict and fills it with the value from varying columns based on the value of the dependend_column in the mapping_dict.

    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(ref_column_1a='value1', ref_column_1b='value3', ref_column_2a='value5', ref_column_2b='value7', dependend_column ='val_1'),
    ...     Row(ref_column_1a='value2', ref_column_1b='value4', ref_column_2a='value6', ref_column_2b='value8', dependend_column ='val_2'),
    ... ])
    >>> dependend_column = 'dependend_column'
    >>> mapping_dict_dict = {'new_column1':{'val_1': 'ref_column_1a','val_2': 'ref_column_1b'}, 'new_column2':{'val_1': 'ref_column_2a','val_2': 'ref_column_2b'}}
    >>> create_new_column_from_varying_columns(df, dependend_column, mapping_dict_dict).show(truncate=False)
    +-------------+-------------+-------------+-------------+----------------+-----------+-----------+
    |ref_column_1a|ref_column_1b|ref_column_2a|ref_column_2b|dependend_column|new_column1|new_column2|
    +-------------+-------------+-------------+-------------+----------------+-----------+-----------+
    |value1       |value3       |value5       |value7       |val_1           |value1     |value5     |
    |value2       |value4       |value6       |value8       |val_2           |value4     |value8     |
    +-------------+-------------+-------------+-------------+----------------+-----------+-----------+
    <BLANKLINE>
    """
    for new_column in mapping_dict_dict:
        df = df.withColumn(new_column, F.lit(None))
        for value, ref_column in mapping_dict_dict[new_column].items():
            df = df.withColumn(
                new_column,
                F.when(df[dependend_column] == value, F.col(ref_column)).otherwise(
                    df[new_column]
                ),
            )

    return df


def keep_latest_in_partition(
    df: DataFrame, partition_column: str, date_column: str
) -> DataFrame:
    """
    Only keep the rows with the newest datetime in the date_column
    for each distinct value in the partition_column.

    >>> spark = getfixture('spark')
    >>> from datetime import datetime as dt
    >>> df = spark.createDataFrame(
    ...     schema=['A', 'B', 'date'],
    ...     data=[
    ...         [1, 1, dt(2020, 1, 1, 0, 0)],
    ...         [1, 2, dt(2020, 1, 2, 0, 0)],  # <- newest 1
    ...         [1, 3, dt(2020, 1, 1, 0, 0)],
    ...         [2, 1, dt(2021, 1, 1, 0, 0)],  # <- newest 2
    ...         [2, 2, dt(2020, 1, 1, 0, 0)],
    ...         [3, 1, dt(2020, 1, 1, 0, 0)],  # <- first 3
    ...         [3, 2, dt(2020, 1, 1, 0, 0)],
    ...         [4, 4, dt(2020, 1, 1, 0, 0)],  # <- only 4
    ...     ]
    ... )
    >>> keep_latest_in_partition(df, 'A', 'date').show(truncate=False)
    +---+---+-------------------+
    |A  |B  |date               |
    +---+---+-------------------+
    |1  |2  |2020-01-02 00:00:00|
    |2  |1  |2021-01-01 00:00:00|
    |3  |1  |2020-01-01 00:00:00|
    |4  |4  |2020-01-01 00:00:00|
    +---+---+-------------------+
    <BLANKLINE>
    """
    window = Window.partitionBy(partition_column).orderBy(F.col(date_column).desc())
    df_with_max_date = df.withColumn(
        f"{date_column}_max", F.max(date_column).over(window)
    )

    return (
        df_with_max_date.filter(F.col(date_column) == F.col(f"{date_column}_max"))
        .drop_duplicates([partition_column])
        .drop(f"{date_column}_max")
    )


def union_all_dataframe(dfs: list[DataFrame]) -> DataFrame:
    """
    Concatenate several pyspark dataframes

    >>> spark = getfixture('spark')
    >>> test_df_1 = spark.createDataFrame([
    ...     Row(col1='1', col2='A', col3=1),
    ... ])
    >>> test_df_2 = spark.createDataFrame([
    ...     Row(col1='2', col2='B', col3=3),
    ... ])
    >>> union_all_dataframe([test_df_1,test_df_2]).show()
    +----+----+----+
    |col1|col2|col3|
    +----+----+----+
    |   1|   A|   1|
    |   2|   B|   3|
    +----+----+----+
    <BLANKLINE>
    """
    return reduce(DataFrame.unionByName, dfs)


def forward_fill(
    df: DataFrame,
    col_to_fill: str,
    order_by: str,
    partition_by: list[str],
    result_col: Optional[str] = None,
    reverse: bool = False,
) -> DataFrame:
    """
    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,0), value=1.0),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,1), value=None),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,2), value=None),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,3), value=2.0),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,4), value=None),
    ... ])
    >>> forward_fill(df, 'value', 'timestamp', [], result_col='transformed_value').show(truncate=False)
    +-------------------+-----+-----------------+
    |timestamp          |value|transformed_value|
    +-------------------+-----+-----------------+
    |2020-01-01 08:00:00|1.0  |1.0              |
    |2020-01-01 08:00:01|null |1.0              |
    |2020-01-01 08:00:02|null |1.0              |
    |2020-01-01 08:00:03|2.0  |2.0              |
    |2020-01-01 08:00:04|null |2.0              |
    +-------------------+-----+-----------------+
    <BLANKLINE>
    """
    if result_col is None:
        result_col = col_to_fill

    partitioned_window = Window.partitionBy(*partition_by)
    order_by_col = F.col(order_by).desc() if reverse else F.col(order_by)
    window_into_the_past = partitioned_window.orderBy(order_by_col).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )

    return df.withColumn(
        result_col,
        F.coalesce(
            F.col(col_to_fill),
            F.last(col_to_fill, ignorenulls=True).over(window_into_the_past),
        ),
    )


def backward_fill(
    df: DataFrame,
    col_to_fill: str,
    order_by: str,
    partition_by: list[str],
    result_col: Optional[str] = None,
) -> DataFrame:
    """
    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,0), value=1.0),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,1), value=None),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,2), value=None),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,3), value=2.0),
    ...     Row(timestamp=datetime.datetime(2020,1,1,8,0,4), value=None),
    ... ])
    >>> backward_fill(df, 'value', 'timestamp', [], result_col='transformed_value').show(truncate=False)
    +-------------------+-----+-----------------+
    |timestamp          |value|transformed_value|
    +-------------------+-----+-----------------+
    |2020-01-01 08:00:04|null |null             |
    |2020-01-01 08:00:03|2.0  |2.0              |
    |2020-01-01 08:00:02|null |2.0              |
    |2020-01-01 08:00:01|null |2.0              |
    |2020-01-01 08:00:00|1.0  |1.0              |
    +-------------------+-----+-----------------+
    <BLANKLINE>
    """
    return forward_fill(
        df, col_to_fill, order_by, partition_by, result_col, reverse=True
    )


def aggregate_sorted_set(
    df: DataFrame, group_by: list[str], col_to_collect: str
) -> DataFrame:
    """
    >>> spark = getfixture('spark')
    >>> df = spark.createDataFrame([
    ...     Row(signal='A', channel='1'),
    ...     Row(signal='A', channel='2'),
    ...     Row(signal='A', channel='2'),
    ...     Row(signal='A', channel='3'),
    ...     Row(signal='B', channel='4'),
    ... ])
    >>> aggregate_sorted_set(df, group_by=['signal'], col_to_collect='channel').show(truncate=False)
    +------+------------+
    |signal|channel_list|
    +------+------------+
    |A     |[1, 2, 3]   |
    |B     |[4]         |
    +------+------------+
    <BLANKLINE>
    """
    return df.groupBy(*group_by).agg(
        F.array_sort(F.collect_set(col_to_collect)).alias(f"{col_to_collect}_list")
    )


def fill_none_values(df):

    w1 = Window.partitionBy("col").orderBy("col")
    w2 = w1.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    cols = [
        "",
        "",
    ]

    return df.select(
        [c for c in df.columns if c not in cols]
        + [
            F.coalesce(F.last(c, True).over(w1), F.first(c, True).over(w2)).alias(c)
            for c in cols
        ]
    )


def create_calendar_weeks_table(supplycase_component):
    min_date = supplycase_component.agg(F.min("date")).head()[0]  # noqa
    max_date = supplycase_component.agg(F.max("date")).head()[0]  # noqa

    return supplycase_component.withColumn(
        "calendar_date",
        F.expr(
            f"explode(sequence(to_date('{min_date}'), to_date('{max_date}'), interval 1 day))"
        ),
    ).withColumn(
        "calendar_week",
        F.concat_ws(
            "",
            F.substring("calendar_date", 1, 4),
            F.format_string("%02d", F.weekofyear("date")),
        ).cast(T.IntegerType()),
    )


def array_union_and_intersect(df):
    """
    Input DF:
    +------------+-----------+
    |arr1        |arr2       |
    +------------+-----------+
    |[a, a, b, c]|[a, d]     |
    |[x, y]      |[i, j]     |
    +------------+-----------+

    Output DF:
    +------------+-----------+------------+--------------+
    |arr1        |arr2       |arr_union   |arr_intersect |
    +------------+-----------+------------+--------------+
    |[a, a, b, c]|[a, d]     |[a, b, c, d]|           [a]|
    |[x, y]      |[i, j]     |[x, y, i, j]|            []|
    +------------+-----------+------------+--------------+

    *arr_union:     distinct elements in either array
    *arr_intersect: elements in both arrays
    """
    return df.withColumns(
        {
            "arr_union": F.array_union("arr1", "arr2"),
            "arr_intersect": F.array_intersect("arr1", "arr2"),
        }
    )


def count_distinct_with_hll(df):
    """
    Input DF:
    +------+------+
    |number|letter|
    +------+------+
    |     1|     a|
    |     2|     a|
    |     3|     a|
    |     4|     b|
    +------+------+
    """
    return df.agg(
        hll_sketch_agg("number").alias("number_hll"),
        hll_sketch_agg("letter").alias("letters_hll"),
    )


def with_columns_renamed(fun):
    def _(df):
        cols = [col(f"`{col_name}`").alias(fun(col_name)) for col_name in df.columns]


def snake_case(df):
    """
    Input DF:
    +--------------+-------------+
    |I like CHEESE |YUMMMY stuff |
    +--------------+-------------+
    |jose          |a            |
    |li            |b            |
    |sam           |c            |
    +--------------+-------------+

    Output DF:
    +--------------+-------------+
    |i_like_cheese |yummmy_stuff |
    +--------------+-------------+
    |jose          |a            |
    |li            |b            |
    |sam           |c            |
    +--------------+-------------+
    """
    return with_columns_renamed(lambda s: s.lower().replace(" ", "_"))(df)


def left_anti_join(df1, df2):
    """
    df1:
    +--+------+
    |id|name  |
    +--+------+
    |1 |socks |
    |2 |chips |
    |3 |ac    |
    |4 |tea   |
    +--+------+

    df2:
    +--+
    |id|
    +--+
    |1 |
    |4 |
    +--+

    Output:
    df1:
    +--+------+
    |id|name  |
    +--+------+
    |2 |chips |
    |3 |ac    |
    +--+------+
    """
    return df1.join(df2, on="id", how="leftanti")
