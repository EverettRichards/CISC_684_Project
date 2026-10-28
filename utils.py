# Downsamples the provided dataframe
import re


def downsample(df,column_name="class",N=1000,random_state=42):
    return (df
    .groupby(column_name, group_keys=False)
    .apply(
        lambda group: group.sample(
            n=min(len(group), N),
            random_state=random_state
        ).assign(**{"class": group.name}),
        include_groups=False
    )
    .reset_index(drop=True)
    .sample(frac=1, random_state=random_state, ignore_index=True)
    )

def squashOneHots(df):
    soil_columns = [f"Soil_Type{i}" for i in range(1,40+1)]
    wilderness_columns = [f"Wilderness_Area{i}" for i in range(1,4+1)]

    df["Soil_Type"] = df[soil_columns].idxmax(axis=1).str[9:]
    df["Wilderness_Area"] = df[wilderness_columns].idxmax(axis=1).str[15:]
    
    df = df.drop(columns=soil_columns).drop(columns=wilderness_columns)
    
    return df

def cleanWhitespace(filePath):
    with open(filePath, "r") as file:
        lines = file.readlines()

    with open(filePath, "w") as file:
        file.writelines(
            re.sub(r"(?<=\S) {2,}", " ", line)
            for line in lines
        )
    