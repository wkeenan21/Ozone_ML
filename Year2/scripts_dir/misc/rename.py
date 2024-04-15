import os
import pandas as pd


def fill_missing_hours(df, datetime_column_name, target_months, constant_columns):
    """
    Fill missing hours in a DataFrame by adding rows for every hour in the range.

    Parameters:
    - df: pandas DataFrame
    - datetime_column_name: str, the name of the datetime column in the DataFrame
    - target_months: list of integers representing months to include
    Returns:
    - DataFrame with missing hours filled, NaN in numeric columns, string in string columns
    """

    # Convert the datetime column to datetime type if it's not already
    df[datetime_column_name] = pd.to_datetime(df[datetime_column_name])
    # Generate a complete hourly range based on the minimum and maximum datetimes in the DataFrame
    complete_range = pd.date_range(start=df[datetime_column_name].min(), end=df[datetime_column_name].max(), freq='H')
    # Create a DataFrame with the complete hourly range
    complete_df = pd.DataFrame({datetime_column_name: complete_range})
    # fill it with the constants too
    for col in constant_columns:
        complete_df[col] = df.reset_index()[col][0]
    # Filter complete DataFrame based on target months
    complete_df = complete_df[complete_df[datetime_column_name].dt.month.isin(target_months)]
    # Merge the complete DataFrame with the existing DataFrame
    merged_df = pd.merge(complete_df, df, on=[datetime_column_name]+constant_columns, how='left')
    return merged_df


for i in range(1,141):

    fold = r'D:\Will_Git\Ozone_ML\Year2\Merged_Data\nh_unaware'
    newName = os.path.join(fold, f'{i}.csv')

    # os.rename(oldName, oldName.replace('_id', ''))
    #
    # newName = oldName.replace('_id', '')
    df = pd.read_csv(newName)

    df = fill_missing_hours(df, 'date', [5,6,7,8,9], ['o3', 'orog', 'pop_den', 'NLCD', 'elev', 'site_name'])
    df.to_csv(newName)



df.index = pd.to_datetime(df['date'])
df['dif'] = df.index.diff()

test = df[df['dif'] != df['dif'].iloc[1]]

pd.infer_freq(df.index)


import os

def count_lines_of_code(directory_path, file_extensions=None):
    """
    Count the total number of lines of code in files within a directory.

    Args:
        directory_path (str): Path to the directory containing the files.
        file_extensions (list, optional): List of file extensions to include. If None, all files will be considered.

    Returns:
        int: Total number of lines of code in the directory.
    """
    total_lines = 0
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Check if the file is a regular file
        if os.path.isfile(file_path):
            # Check if file extension matches if specified
            if file_extensions is None or filename.endswith(tuple(file_extensions)):
                # Count lines in the file
                with open(file_path, 'r') as file:
                    total_lines += sum(1 for line in file)
    return total_lines

# Example usage
fold = r'D:\Will_Git\Ozone_ML\Year2\scripts_dir'
total = 0
for file in os.listdir(fold):
    if os.path.isdir(os.path.join(fold, file)):
        directory_path = os.path.join(fold, file)
        total = total+count_lines_of_code(directory_path, ['.py'])

print(total)

