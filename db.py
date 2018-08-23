#!/usr/bin/python
import psycopg2
import os

db_conf = {
    "host": os.environ['DB_HOST'],
    "user": os.environ['DB_USER'],
    "password": os.environ['DB_PASSWORD'],
    "dbname": os.environ['DB_NAME']
}

def put(table=None, value=''):
    """ Inserts a value in some table """
    sql = """INSERT INTO {table} VALUES ({value})""".format(table=table, value=value)
    with psycopg2.connect(**db_conf) as con:
        with con.cursor() as cur:
            cur.execute(sql, {'table': table, 'value': value})

def get(sql):
    """ Returns rows using sql """
    #sql = """INSERT INTO public.numbers VALUES (%(n)s)"""
    with psycopg2.connect(**db_conf) as con:
        with con.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return rows