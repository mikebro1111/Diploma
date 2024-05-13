from datetime import datetime, timedelta

import pandas as pd
import dateutil.parser


class DataService:

    def __init__(self, date_format='%Y-%m-%d'):
        self.date_format = date_format

    def process_data(self, file_path):
        read_data = pd.read_json(file_path)
        read_last_date_df = str(read_data['Date'].values[-1])
        last_date_df = dateutil.parser.isoparse(read_last_date_df)

        today = datetime.today()
        dif_day_dates = int(today.strftime('%d')) - int(last_date_df.strftime('%d'))

        start = datetime.today() - last_date_df
        print(start)
        print(type(start))

        start = (last_date_df + timedelta(days=1)).strftime(self.date_format)
        end = datetime.now().strftime(self.date_format)
        if dif_day_dates > 0:
            read_data.to_json(file_path)  # Відносний шлях

        return start, end

