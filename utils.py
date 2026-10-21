# Downsamples the provided dataframe
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
    )