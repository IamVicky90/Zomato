from src.logger import logger
import sqlite3
import json
import os
import csv
import pandas as pd


class db_ops:
    def __init__(self):
        self.log = logger.log()
        self.json_file = json.load(open('schema_prediction.json',))

    def con_establish(self, dbname):
        self.log.log_writer(
            f'Establishing Connection with {dbname} ....', 'DB_Operations.log', service_name='p')
        return sqlite3.connect(dbname)

    def create_Table(self, Table_name, dbname):
        if dbname in os.listdir(os.getcwd()):
            os.remove(dbname)  # we are removing the existing database
            self.log.log_writer(
                f'DataBase {dbname} removed', 'DB_Operations.log', 'Warning', service_name='p')
        try:
            conn = self.con_establish(dbname)
            self.log.log_writer(
                f'Successfully Established Connection with {dbname}', 'DB_Operations.log', service_name='p')
        except Exception as e:
            self.log.log_writer(
                f'Could not Established Connection with {dbname} error: {str(e)}', 'DB_Operations.log', 'Error', service_name='p')
        cur = conn.cursor()
        i = 0
        try:
            for col_name, d_type in self.json_file['ColName'].items():
                if i == 0:  # To avoid iterations
                    cur.execute(
                        f'Create Table if not exists {Table_name} ({col_name} {d_type})')
                    conn.commit()
                    i = i+1
                if col_name == 'listed_in(type)':
                    # Because Brackets are not allowed in column name in sqllite
                    col_name = 'listed_in_type'
                if col_name == "listed_in(city)":
                    # Because Brackets are not allowed in column name in sqllite
                    col_name = "listed_in_city"
                if col_name == "approx_cost(for two people)":
                    # Because Brackets are not allowed in column name in sqllite
                    col_name = "approx_cost_for_two_people"

                try:
                    cur.execute(
                        f'ALter Table {Table_name} Add {str(col_name)} {d_type}')
                    conn.commit()
                except sqlite3.OperationalError:
                    continue
                except Exception as e:
                    self.log.log_writer(
                        f'Could not create the table or column error: {str(e)}', 'DB_Operations.log', 'Error', service_name='p')

            self.log.log_writer(
                f"Successfully created the {Table_name} along with it's columns", 'DB_Operations.log', service_name='p')
        except Exception as e:
            self.log.log_writer(
                f'Could not create the table or column error: {str(e)}', 'DB_Operations.log', 'Error', service_name='p')

        conn.close()

    def insert_values_into_table(self, dbname, table):
        try:
            conn = self.con_establish(dbname)
            cur = conn.cursor()
            for file in os.listdir('Prediction_Good_Data_Folder'):
                with open(os.path.join('Prediction_Good_Data_Folder', file),) as csvfile:
                    rows = csv.reader(csvfile)
                    next(rows)
                    for row in rows:
                        cur.execute(
                            f"INSERT INTO {table} VALUES (?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row[1:])
                    conn.commit()
            self.log.log_writer(
                f'Sucessfully inserted the values into table {table}', 'DB_Operations.log', service_name='p')
        except Exception as e:
            self.log.log_writer(
                f'Could not insert values into table {table} error: {str(e)}', 'DB_Operations.log', 'Error', service_name='p')

    # To create a master file of this sqllite database
    def dump_data_from_database_to_one_csv_file(self, dbname, table):
        try:
            conn = self.con_establish(dbname)
            cur = conn.cursor()
            csv_file = pd.read_sql_query(f'select * from {table}', conn)
            os.makedirs('Master_Prediction_File', exist_ok=True)
            csv_file.to_csv(os.path.join('Master_Prediction_File',
                            'Zomato_prediction.csv'), index=False)
            self.log.log_writer(
                f'Successfully dump the data from {dbname} database to one csv file', 'DB_Operations.log', service_name='p')
        except Exception as e:
            self.log.log_writer(
                f'Could not  dump the data from {dbname} database to one csv file error: {str(e)}', 'DB_Operations.log', 'Error', service_name='p')
