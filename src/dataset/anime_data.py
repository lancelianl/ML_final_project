import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

class AnimeData:
    def __init__(self, filepath='../../data/raw/anime.csv'):
        self.filepath = filepath
        self.df = None
        self._load_and_preprocess()
        
    def _load_and_preprocess(self):
        # Load the data
        df = pd.read_csv(self.filepath)
        
        # Replace 'Unknown' with NaN for easier handling
        df = df.replace('Unknown', np.nan)
        
        # Convert Genres from comma-separated strings into lists (if not NaN)
        df['Genres'] = df['Genres'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
        
        # Parse the Aired column
        start_dates, end_dates = [], []
        for aired in df['Aired']:
            s_date, e_date = self._parse_aired(aired)
            start_dates.append(s_date)
            end_dates.append(e_date)
        df['StartDate'] = start_dates
        df['EndDate'] = end_dates

        # Mean-center the dates
        valid_dates = df[['StartDate', 'EndDate']].values.flatten()
        valid_dates = valid_dates[~np.isnan(valid_dates)]
        if len(valid_dates) > 0:
            date_mean = np.mean(valid_dates)
            df['StartDate'] = df['StartDate'] - date_mean
            df['EndDate'] = df['EndDate'] - date_mean

        # Convert Score to float
        if 'Score' in df.columns:
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

        self.df = df

    def _parse_aired(self, aired_str):
        if pd.isna(aired_str):
            return np.nan, np.nan
        parts = aired_str.split(' to ')
        start_date = self._parse_single_date(parts[0].strip())
        if len(parts) > 1:
            end_date_str = parts[1].strip()
            if end_date_str == '?':
                end_date = self._float_from_ymd(2025, 1, 1)
            else:
                end_date = self._parse_single_date(end_date_str)
        else:
            # No end date, assume same as start
            end_date = start_date
        return start_date, end_date

    def _parse_single_date(self, date_str):
        if date_str == '?':
            return self._float_from_ymd(2025, 1, 1)
        pattern = r'([A-Za-z]+)\s*(\d{1,2})?,?\s*(\d{4})'
        match = re.search(pattern, date_str)
        if not match:
            return np.nan
        month_str = match.group(1)
        day_str = match.group(2)
        year_str = match.group(3)

        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }

        month = month_map.get(month_str[:3], np.nan)
        if pd.isna(month):
            return np.nan

        if day_str is None:
            day = 15  # default day if not provided
        else:
            day = int(day_str)
        year = int(year_str)

        return self._float_from_ymd(year, month, day)

    def _float_from_ymd(self, year, month, day):
        return year + (month / 12.0) + (day / 365.0)

    def get_anime_by_id(self, mal_id):
        row = self.df[self.df['MAL_ID'] == mal_id]
        if row.empty:
            return None
        return row.to_dict('records')[0]

    def get_anime_by_name(self, name):
        matches = self.df[self.df['Name'] == name]
        return matches.to_dict('records') if not matches.empty else []

    def list_columns(self):
        return self.df.columns.tolist()
        
    def save_processed(self, outpath='data/processed/anime_processed.csv'):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        self.df.to_csv(outpath, index=False)

    def genre_mean_score(self, genres):
        """
        Compute the mean score of all anime that have at least one of the provided genres.
        genres: list of genre strings (e.g., ['Comedy', 'Drama'])
        
        We consider the union of these genres: if an anime has at least one of the genres, it qualifies.
        """
        # Ensure genres is a list
        if not isinstance(genres, list):
            genres = [genres]

        # Filter rows that contain at least one of the genres
        mask = self.df['Genres'].apply(lambda g_list: any(genre in g_list for genre in genres))
        subset = self.df[mask]
        
        if subset.empty:
            return None

        # Compute the mean of the Score ignoring NaN
        mean_score = subset['Score'].mean(skipna=True)
        
        return mean_score if not np.isnan(mean_score) else None


# Example usage:
if __name__ == "__main__":
    anime_data = AnimeData(filepath='../../data/raw/anime.csv')
    # Save processed data
    anime_data.save_processed()
    
    # Compute genre mean score
    genres_to_check = ['Comedy', 'Drama']
    print("Mean score for genres {}: {}".format(genres_to_check, anime_data.genre_mean_score(genres_to_check)))
