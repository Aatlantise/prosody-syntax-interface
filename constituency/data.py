import pandas as pd

def load_data(filepath):
    """
    Tabular prosody data to df

    :param filepath:
    :return: dataframe
    """
    df = pd.read_csv(filepath, sep='\t', names=["start", "end", "token"], keep_default_na=False)
    return df

def extract_examples_from_sent(df):
    """
    Df to dictionary, including token, pause, and duration data

    :param df:
    :return:
    """
    current_context = []

    for i, row in df.iterrows():
        start, end, token = row["start"], row["end"], row["token"]
        if not token: # do not consider pause at this time
            if current_context:
                current_context[-1]["pause"] = row['end'] - row['start']
        else:
            current_context.append({
                "start": float(start),
                "end": float(end),
                "token": token,
                "pause": 0. # to be corrected in the following pause token when applicable
            })

    # Flatten into list of examples
    examples = []
    max_duration = 0.
    max_pause = 0.
    for i in range(len(current_context)):
        duration = current_context[i]["end"] - current_context[i]["start"]
        pause = current_context[i]["pause"]
        max_duration = max(max_duration, duration)
        max_pause = max(max_pause, pause)
        examples.append({
            "text": current_context[i]["token"],
            "duration": duration,
            "pause": pause
        })
    return examples, max_duration, max_pause
