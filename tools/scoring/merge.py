import argparse

import pandas as pd
from glob import glob


def merge(args):
    df = pd.read_csv(args.meta_path)

    id2res = {}
    for path in sorted(glob(f'{args.part_dir}/*{args.column}*.csv')):
        df_part = pd.read_csv(path)
        for _, row in df_part.iterrows():
            id2res[row['id']] = row[args.column]
    
    df[args.column] = [id2res[id] for id in df['id'].tolist()]
    save_path = args.meta_path.replace('.csv', f'_{args.column}.csv')
    df.to_csv(save_path, index=False)
    print(f'saved {len(df)} {args.column} results to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--part_dir", type=str, help="Directory to partial results")
    parser.add_argument("--column", type=str, help="Column name of score type")
    args = parser.parse_args()

    merge(args)
