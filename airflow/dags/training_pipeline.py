# airflow/dags/training_pipeline.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {'owner': 'airflow', 'start_date': days_ago(1)}

with DAG(
    dag_id='house_price_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python /opt/airflow/src/preprocess.py'
    )

    train = BashOperator(
        task_id='train',
        bash_command='python /opt/airflow/src/train.py'
    )

    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='python /opt/airflow/src/evaluate.py'
    )

    preprocess >> train >> evaluate